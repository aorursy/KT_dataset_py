VERSION = "nightly"  #@param ["1.5" , "20200325", "nightly"]
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION 
import torch 
import transformers
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl 
import torch_xla.distributed.xla_multiprocessing as xmp
from tqdm import tqdm
import torch.nn as nn 
from sklearn import metrics, model_selection
import numpy as np
import pandas as pd 
from transformers import AdamW, get_linear_schedule_with_warmup 
class BERTDataset:
    def __init__(self, premise, hypothesis,label):
        self.premise = premise
        self.hypothesis = hypothesis
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        self.label = label

    def __len__(self):
        return len(self.premise)

    def __getitem__(self, item):
        premise = str(self.premise[item])
        hypothesis = str(self.hypothesis[item])
        inputs = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label': torch.tensor(self.label[item], dtype=torch.float)            
        }


class BERTBaseMultilingualCased(nn.Module):
    def __init__(self):
        super(BERTBaseMultilingualCased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            BERT_PATH
        )
        self.bert_drop = nn.Dropout(0.3)
        
        self.out = nn.Linear(768, 3)


    def forward(self, ids, mask, token_type_ids):
        _, out2 = self.bert(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
        ) 
        bo = self.bert_drop(out2) 
        output = self.out(bo)

        return output


def loss_fn(output, target):
    return nn.CrossEntropyLoss()(output, target)


def train_fn(data_loader, model, device, optimizer,scheduler):
    model.train()
    for b, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        target = d['label']
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        target = target.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(ids, mask, token_type_ids)
        loss = loss_fn(output, target)
        loss.backward()
        xm.optimizer_step(optimizer)
        if scheduler is not None:
          scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_target = []
    fin_output = []
    with torch.no_grad():
        for b, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d['ids']
            mask = d['mask']
            token_type_ids = d['token_type_ids']
            target = d['label']
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.long)
            output = model(ids, mask, token_type_ids)
            output = torch.log_softmax(output,dim=1)
            output = torch.argmax(output,dim=1)
            fin_target.extend(target.cpu().detach().numpy().tolist())
            fin_output.extend(output.cpu().detach().numpy().tolist())
    return fin_target, fin_output 

EPOCHS = 10
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
BERT_PATH = 'bert-base-multilingual-cased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=False)
MAX_LEN = 260
MODEL_PATH = './model.bin'
TRAINING_FILE = '/kaggle/input/contradictory-my-dear-watson/train.csv'
TESTING_FILE = '/kaggle/input/contradictory-my-dear-watson/test.csv'

def run():
    dfx = pd.read_csv(TRAINING_FILE).fillna('none')
    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, stratify=dfx.label.values, random_state=42)
    
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    

    def get_dataset():
        train_dataset = BERTDataset(
          df_train.premise.values, df_train.hypothesis.values, df_train.label.values)
        valid_dataset = BERTDataset(
          df_valid.premise.values, df_valid.hypothesis.values, df_valid.label.values)  
        return train_dataset,valid_dataset
    SERIAL_EXEC = xmp.MpSerialExecutor()
    train_dataset,valid_dataset = SERIAL_EXEC.run(get_dataset)

    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas = xm.xrt_world_size(),
        rank = xm.get_ordinal(),
        shuffle = True 
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler = train_sampler)
     
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas = xm.xrt_world_size(),
        rank = xm.get_ordinal(),
        shuffle = True 
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE,sampler = valid_sampler) 
    
    MODEL_WRAPPER = xmp.MpModelWrapper(BERTBaseMultilingualCased())
    device = xm.xla_device()
    model = MODEL_WRAPPER.to(device)
    LR = 3e-5*xm.xrt_world_size()
    num_steps = int(len(train_dataset)/TRAIN_BATCH_SIZE/xm.xrt_world_size() * EPOCHS)
    optimizer = AdamW(model.parameters(),lr=LR)  
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_steps
    )
    best_accuracy = 0
    for EPOCH in range(EPOCHS): 
        para_train_loader = pl.ParallelLoader(train_data_loader,[device])
        train_fn(para_train_loader.per_device_loader(device), model, device, optimizer,scheduler=scheduler)
        para_valid_loader = pl.ParallelLoader(valid_data_loader, [device])
        target, output = eval_fn(para_valid_loader.per_device_loader(device), model, device)
        accuracy = metrics.accuracy_score(target, output)
        xm.master_print(f'{EPOCH+1}: Accuracy Score: {accuracy}')
        if accuracy >= best_accuracy:
            xm.save(model.state_dict(), MODEL_PATH)
            best_accuracy = accuracy
    return f'Best Accuracy: {best_accuracy}'
xmp.spawn(run(),nprocs=8) 
import numpy
class BERTInferenceDataset:
    def __init__(self, premise, hypothesis):
        self.premise = premise
        self.hypothesis = hypothesis
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.premise)

    def __getitem__(self, item):
        premise = str(self.premise[item])
        hypothesis = str(self.hypothesis[item])
        inputs = self.tokenizer.encode_plus(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)           
        }

import torch 
from tqdm import tqdm
import torch.nn as nn  
import numpy as np
import pandas as pd 
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup 
TESTING_FILE = '/kaggle/input/contradictory-my-dear-watson/test.csv'
BERT_PATH = 'bert-base-multilingual-cased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=False)
MAX_LEN = 260
MODEL_PATH = './model.bin'
df = pd.read_csv(TESTING_FILE).fillna('none')
test_dataset = BERTInferenceDataset(df.premise,df.hypothesis)
test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 8)
model = BERTBaseMultilingualCased() 
device = 'cpu'
model.load_state_dict(torch.load(MODEL_PATH)) 
model.to(device)
fin_output = []
model.eval()
with torch.no_grad():
    for b, d in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
        ids = d['ids']
        mask = d['mask']
        token_type_ids = d['token_type_ids']
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long) 
        output = model(ids, mask, token_type_ids)
        output = torch.log_softmax(output,dim=1) 
        output = torch.argmax(output,dim=1)
        fin_output.extend(output.cpu().detach().numpy().tolist())
submission = pd.DataFrame({'id': df['id'], 'prediction': fin_output}) 
submission.to_csv('submission.csv', index=False) 
