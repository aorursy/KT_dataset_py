import gc

import os

import pandas as pd

import numpy as np



import random



import torch

import torch.nn as nn

from torch.nn import functional as F

from torch.utils.data import DataLoader



from tqdm.autonotebook import tqdm

import pickle



from transformers import *



from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score



device = torch.device('cuda')
class config:

    TOKEN_ID_DIR = "/kaggle/input/aio-make-token-ids"

    SEED = 416

    TRAIN_BATCH_SIZE = 2

    VALID_BATCH_SIZE = 2

    OPTIONS = 4

    EPOCHS = 1

    LEARNING_RATE = 1e-6

    MODEL_TYPE = "cl-tohoku/bert-base-japanese"
with open(f"{config.TOKEN_ID_DIR}/train.pkl", "rb") as f:

    train = pickle.load(f)

with open(f"{config.TOKEN_ID_DIR}/dev1.pkl", "rb") as f:

    dev1 = pickle.load(f)

with open(f"{config.TOKEN_ID_DIR}/dev2.pkl", "rb") as f:

    dev2 = pickle.load(f)
class JaSQuADBert(nn.Module):

    def __init__(self):

        super(JaSQuADBert, self).__init__()



        self.bert = BertModel.from_pretrained(config.MODEL_TYPE)

        self.qa_outputs = nn.Linear(768, 2)



    

    def forward(self, ids, mask, token_type_ids):

        out, _ = self.bert(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )

        logits = self.qa_outputs(out)

        

        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits

    

jasquad_model = JaSQuADBert()

jasquad_model.load_state_dict(torch.load("/kaggle/input/jasquad-train-bert/model_2.bin", map_location=torch.device('cpu')))

torch.save(jasquad_model.bert.state_dict(), "squad_weight.bin")

del jasquad_model

!ls
class BertForAIO(nn.Module):

    def __init__(self):

        super(BertForAIO, self).__init__()



        self.bert = AutoModel.from_pretrained(config.MODEL_TYPE)

        self.bert.load_state_dict(torch.load("squad_weight.bin", map_location=torch.device('cpu')))

        

        self.fc = nn.Linear(768, 1)



        

    def forward(self, ids, mask, token_type_ids):

        n_choice = ids.shape[1]

        

        ids = ids.view(-1, ids.size(-1))

        mask = mask.view(-1, mask.size(-1))

        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))



        _, h = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        logits = self.fc(h)

        logits = logits.view(-1, n_choice)



        return logits
class JaqketDataset:

    def __init__(self, data, train=True):

        self.data = data

        self.train = train

    

    def __len__(self):

        return len(self.data)



    def __getitem__(self, item):

        d = self.data[item]

        if self.train:

            return {'ids': torch.tensor(d["input_ids"][:config.OPTIONS], dtype=torch.long),

                    'mask': torch.tensor(d["attention_mask"][:config.OPTIONS], dtype=torch.long),

                    'token_type_ids': torch.tensor(d["token_type_ids"][:config.OPTIONS], dtype=torch.long),

                    'targets': torch.tensor(d["label"], dtype=torch.long)}

        else:

            return {'ids': torch.tensor(d["input_ids"], dtype=torch.long),

                    'mask': torch.tensor(d["attention_mask"], dtype=torch.long),

                    'token_type_ids': torch.tensor(d["token_type_ids"], dtype=torch.long),

                    'targets': torch.tensor(d["label"], dtype=torch.long)}
train_dataset = JaqketDataset(train, train=True)

train_data_loader = DataLoader(train_dataset,

                               batch_size=config.TRAIN_BATCH_SIZE,

                               shuffle=True,

                               drop_last=True,

                               num_workers=2)



dev1_dataset = JaqketDataset(dev1, train=False)

dev1_data_loader = DataLoader(dev1_dataset,

                              batch_size=config.TRAIN_BATCH_SIZE,

                              shuffle=False,

                              drop_last=False,

                              num_workers=2)



dev2_dataset = JaqketDataset(dev2, train=False)

dev2_data_loader = DataLoader(dev2_dataset,

                              batch_size=config.TRAIN_BATCH_SIZE,

                              shuffle=False,

                              drop_last=False,

                              num_workers=2)
model = BertForAIO()

model.to(device)

optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
trn_losses = []

for epoch in range(config.EPOCHS):

    # 学習

    model.train()

    for d in tqdm(train_data_loader):

        ids = d["ids"].to(device, dtype=torch.long)

        mask = d["mask"].to(device, dtype=torch.long)

        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

        targets = d["targets"].to(device, dtype=torch.long)

        

        model.zero_grad()

        y_pred = model(ids, mask, token_type_ids)

        loss = nn.CrossEntropyLoss()(y_pred, targets)

        loss.backward()

        optimizer.step()

        trn_losses.append(loss.item())

        

    # 評価

    model.eval()

    dev1_scores, dev2_scores = [], []

    with torch.no_grad(): 

        for d in tqdm(dev1_data_loader):

            ids = d["ids"].to(device, dtype=torch.long)

            mask = d["mask"].to(device, dtype=torch.long)

            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

            targets = d["targets"].to(device, dtype=torch.long).cpu().numpy()



            y_pred = model(ids, mask, token_type_ids)

            y_pred = y_pred.cpu().detach().numpy().argmax(1)

            acc = accuracy_score(targets, y_pred)

            dev1_scores.append(acc)

        dev1_acc = sum(dev1_scores)/len(dev1_scores)

        

        for d in tqdm(dev2_data_loader):

            ids = d["ids"].to(device, dtype=torch.long)

            mask = d["mask"].to(device, dtype=torch.long)

            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

            targets = d["targets"].to(device, dtype=torch.long).cpu().numpy()



            y_pred = model(ids, mask, token_type_ids)

            y_pred = y_pred.cpu().detach().numpy().argmax(1)

            acc = accuracy_score(targets, y_pred)

            dev2_scores.append(acc)

        dev2_acc = sum(dev2_scores)/len(dev2_scores)



    print(f"{epoch} epoch: dev1={dev1_acc} / dev2={dev2_acc}")

    torch.save(model.state_dict(), f"aio_bert_epoch_{epoch}.bin")
print(f"{epoch} epoch: dev1={dev1_acc} / dev2={dev2_acc}")
plt.plot(trn_losses)
del train, train_dataset, train_data_loader

del dev1, dev1_dataset, dev1_data_loader

del dev2, dev2_dataset, dev2_data_loader

gc.collect()
df_aio_leaderboard = pd.read_json(f"{config.TOKEN_ID_DIR}/aio_leaderboard.json", orient='records', lines=True)       



with open(f"{config.TOKEN_ID_DIR}/test.pkl", "rb") as f:

    test = pickle.load(f)

    

test_dataset = JaqketDataset(test, train=False)
predicts = []

model.eval()

with torch.no_grad():

    for idx, d in enumerate(test_dataset):

        ids = d["ids"].to(device, dtype=torch.long).unsqueeze(0)

        mask = d["mask"].to(device, dtype=torch.long).unsqueeze(0)

        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long).unsqueeze(0)

        

        y_pred = model(ids, mask, token_type_ids)

        y_pred = y_pred.cpu().detach().numpy().argmax(1)



        p = {"qid": df_aio_leaderboard.loc[idx, "qid"],

             "answer_entity": df_aio_leaderboard.loc[idx, "answer_candidates"][y_pred[0]]}

        predicts.append(p)



pd.DataFrame(predicts).to_json(f'lb_predict.jsonl', orient='records', force_ascii=False, lines=True)
!head "lb_predict.jsonl"