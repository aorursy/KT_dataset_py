import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import transformers
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "../input/bert-base-uncased/pytorch_model.bin"
TRAINING_FILE = "../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
class BertBaseUncased(nn.Module):
    def __init__(self):
        super (BertBaseUncased,self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.drop_out = nn.Dropout(0.2)
        self.out = nn.Linear(768,1)
        
    def forward(self,ids,mask,token_type_ids):
        _,o2 = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        bo = self.drop_out(o2)
        output = self.out(bo)
        
        return output
        
class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
        
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self,item):
        review = str(self.review[item])
        review = " ".join(review.split())
        
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True)
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        
        return {
            "ids" : torch.tensor(ids, dtype=torch.long),
            "mask" : torch.tensor(mask,dtype=torch.long),
            "token_type_ids" : torch.tensor(token_type_ids,dtype=torch.long),
            "targets" : torch.tensor(self.target[item], dtype=torch.float)
        }
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

def train_fn(data_loader, model, optimizer, device, schdeular):
    model.train()
    
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        optimizer.step()
        schdeular.step()
        
def eval_fun(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    
    with torch.no_grad(): 
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]
        
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
        
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
df = pd.read_csv(TRAINING_FILE).fillna("none")
df.head()
df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.sentiment.values)
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

train_dataset = BERTDataset(review = df_train.review.values, target = df_train.sentiment.values)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size= TRAIN_BATCH_SIZE, num_workers = 4)

valid_dataset = BERTDataset(review = df_valid.review.values, target = df_valid.sentiment.values)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers = 1)
device = torch.device(DEVICE)
model = BertBaseUncased()
model.to(device)
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


num_train_steps = int(len(df_train)) / TRAIN_BATCH_SIZE * EPOCHS
optimizer = AdamW(optimizer_parameters, lr = 3e-5)
schedular = get_linear_schedule_with_warmup(
                optimizer,num_warmup_steps=0,num_training_steps=num_train_steps)

best_accurcy = 0

for epoch in range(2):
    train_fn(train_data_loader, model, optimizer, device, schedular)
    outputs, targets = eval_fun(valid_data_loader, model, device)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    print(f"Accuracy Score = {accuracy}")
        

