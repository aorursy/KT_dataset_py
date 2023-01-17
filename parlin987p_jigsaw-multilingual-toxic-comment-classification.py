!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import transformers
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from tqdm import tqdm

#for tpus to run on multiple device
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train =  pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train.head()
MAX_LEN = 128
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 64
EPOCHS = 10
BERT_PATH = "../input/bert-base-multilingual-uncased/"
MODEL_PATH = "model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
                                                        BERT_PATH,
                                                        do_lower_case=True )
#dataset - dataloader
class BERTDataset:
    def __init__(self, comment_text, target):
        self.comment_text = comment_text
        self.target = target
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
    
    def __len__(self):
        return len(self.comment_text)
    
    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        padding_length = self.max_length - len(ids)        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float)
        }

#model
class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        #taking the pretrained tranformers model
        self.bert_path = BERT_PATH
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        #we want one input, in bert model in the last layer is getting 768 parameters
        self.out = nn.Linear(768, 1)

   #this petrained model take some parameters as below
    def forward(self, ids, mask, token_type_ids):
        o1, o2 = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_types_ids
        )
        #dropout 
        bo = self.bert_drop(o2)
        #linear as above 
        output = self.out(bo)
        print("o1,o2",o1, o2)
        return output

#engine
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
#go over all the batches in the dataloader
    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
#put into device cuda
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
#pass everything to the model
        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
#calculate the loss
        loss = loss_fn(outputs, targets)
        if bi % 10 == 0:
            xm.master_print(f'bi={bi}, loss={loss}')
#backward prop
        loss.backward()
#step the optimizer and scheduler 
        xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()

#the same as above without loss part
def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
#for tpus we don't need no_grad, we removed tqdm()
    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
#train
def _run():
    mx = BERTBaseUncased(bert_path="../input/bert-base-multilingual-uncased/")
    
    df_train_bias = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text","toxic"] )
    df_train_base = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text","toxic"] )
    df_train = pd.concat([df_train_base,df_train_bias],axis=0).reset_index()

    df_valid = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

    #creating a datset object
    train_dataset = BERTDataset(
        comment_text=df_train.comment_text.values,
        target=df_train.toxic.values
    )
    
    #for TPUs - to load on different nodes
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                        train_dataset, 
                                                        num_replicas= xm.xrt_world_size() ,
                                                        rank =xm.get_ordinal() ,
                                                        shuffle = True
    )
    
    #creating a datset loader object
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=4, 
        #for tpus
        sampler = train_sampler,
        drop_last=True
    )

    valid_dataset = BERTDataset(
        comment_text=df_valid.comment_text.values,
        target=df_valid.toxic.values
    )
    
        #for TPUs - to load on different nodes
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
                                                        train_dataset, 
                                                        num_replicas= xm.xrt_world_size() ,
                                                        rank =xm.get_ordinal() ,
                                                        shuffle = True
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=1,
         #for tpus
        sampler = valid_sampler,
        drop_last=False
    )
#tpus - made a change
    device = xm.xla_device()
    #creating a model
#     model = xm.to(device)
    model= mx.to(device)
    
    #don't touch it unless we want to improve score, variation with text len etc... 
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    #below - I don;t want wd for layers with the name above
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    #tpus - made a change below
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE / xm.xrt_world_size() *  config.EPOCHS)
    lr = 3e-5 * xm.xrt_world_size()
    
    #Creates an optimizer with learning rate schedule.
#     optimizer.zero_grad()
    optimizer = AdamW(optimizer_parameters, lr=lr)
    #Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period.    there are several options at https://huggingface.co/transformers/main_classes/optimizer_schedules.html
    scheduler = get_linear_schedule_with_warmup(
        #made changed
        xm.optimizer_step(optimizer, barrier=True),
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(EPOCHS):
        #train
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        train_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler)
        #valid
        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)
        print('outputs', o , 'targets' ,t)
        xm.save(model.state_dict(), "model.bin")
        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)
        xm.master_print(f'AUC = {auc}')     
#         if accuracy > best_accuracy:
#             #made a change too
#             xm.save(model.state_dict(), config.MODEL_PATH)
#             best_accuracy = accuracy

# Start training processes
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a =_run()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
