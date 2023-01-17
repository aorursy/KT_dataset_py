!nvidia-smi
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import gc

import glob

import torch

from tqdm.autonotebook import tqdm

from transformers import *

from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

from sklearn.model_selection import KFold
import torch

from torch import nn

from torch.utils.data import DataLoader, random_split

from torch.nn import functional as F

from torchvision.datasets import MNIST

from torchvision import datasets, transforms

import logging
logging.info("Imports Complete")
ROOT_DIR =  "../input/tweet-sentiment-extraction/"
train = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))

test = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))

sub = pd.read_csv(os.path.join(ROOT_DIR, "sample_submission.csv"))
train.head()
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

#model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased')
!ls ../input/bert-qa/
tokenizer = BertTokenizer.from_pretrained('../input/bert-qa/tokenizer/')

#model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')



model = BertForQuestionAnswering.from_pretrained('../input/bert-qa//model/')
train.dropna(0, how="any", inplace=True)

train = train.reset_index()
train["sentiment_q"]=train["sentiment"].apply(lambda x: "what text is "+x+"?")

test["sentiment_q"]=test["sentiment"].apply(lambda x: "what text is "+x+"?")
train
def get_beg_end(full_str, sub_str):

        start = full_str.index(sub_str)

        end = start+len(sub_str)

        return (start, end)

vfunc = np.vectorize(get_beg_end)    
train["start"], train["end"] = vfunc(train["text"].values, train["selected_text"].values)


class TextDataset(torch.utils.data.Dataset):



    def __init__(self, data, tokenizer,is_test=False, return_attention_masks=False, max_length=200):

        self.df = data

        self.tokenizer = tokenizer

        self.is_test = is_test

        self.return_attn_mask = return_attention_masks

        self.max_len = max_length



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        #print(idx)

        #new_list = [(a, b) for a,b in zip(self.df.loc[idx, "sentiment_q"], self.df.loc[idx, "text"])]

        s, t = self.df.loc[idx, "sentiment_q"], self.df.loc[idx, "text"]

        #print(s, t)

        token_ids = self.tokenizer.encode_plus(s, t, add_special_tokens=True,

                                                     return_attention_mask=self.return_attn_mask,

                                                     max_length = self.max_len,

                                                     pad_to_max_length=True

                                                     )

        input_ids = torch.tensor(token_ids["input_ids"])

        attn_mask = torch.tensor(token_ids["attention_mask"])



        #print(torch.tensor(token_ids["input_ids"]).shape)

        #print(token_ids)

        if self.is_test:

            return input_ids, attn_mask

        return input_ids, attn_mask, self.df.loc[idx, "start"], self.df.loc[idx, "end"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

kf.get_n_splits(train)
max_len = len(max(train["text"].values, key=len))

print(max_len+len("What text is neutral?"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

critation = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



jac_v = np.vectorize(jaccard)
device = torch.device("cuda")
model.to(device);
train_loss_hist = []

val_loss_hist = []

trn_epoch_loss=[]

val_epoch_loss =[]
def train_fn(model, train_loader, optimizer):

    print("training...")

    train_loss=0

    for i, train_batch in tqdm(enumerate(train_loader), total=len(train_set)//batch_size):

        op = train_batch

        input_ids = op[0].to(device)

        attn_masks = op[1].to(device)

        start_t = op[2].to(device)

        end_t = op[3].to(device)



        optimizer.zero_grad()



        start_p, end_p = model(input_ids, attention_mask=attn_masks)



        loss = (critation(start_p, start_t) + critation(end_p, end_t))/2.0

        train_loss_hist.append(loss.item()/batch_size)

        if i%300 ==0:

            print(f"[{i}, loss: {loss.item()/batch_size}]")

        

        loss.backward()

        optimizer.step



        torch.cuda.empty_cache()

        gc.collect()

    return np.mean(train_loss_hist)
def evaluate(model, valid_loader):

    val_s=[]

    val_e=[]

    with torch.no_grad():

        for i, valid_batch in tqdm(enumerate(valid_loader), total = len(valid_set)//batch_size):

            op = valid_batch

            input_ids = op[0].to(device)

            attn_masks = op[1].to(device)

            start_t = op[2].to(device)

            end_t = op[3].to(device)



            start_p, end_p = model(input_ids, attention_mask=attn_masks)

            val_s.append(torch.argmax(start_p, 1).cpu())

            val_e.append(torch.argmax(end_p, 1).cpu())



            loss = (critation(start_p, start_t) + critation(end_p, end_t))/2.0

            val_loss_hist.append(loss.item()/batch_size)

            if i%100 ==0:

                print(f"[{i}, loss: {loss.item()}]")



        assert torch.cat(val_s).shape[0] == val_.shape[0]

        val_s = torch.cat(val_s)

        assert torch.cat(val_e).shape[0] == val_.shape[0]

        val_e = torch.cat(val_e)



        val_["predicted_text"]=""

        for num in (range(val_.shape[0])):

            val_.loc[num, "predicted_text"] = val_.loc[num, "text"][val_s[num]:val_e[num]]

        scores = jac_v(val_["selected_text"].values, val_["predicted_text"].values)

        print(f"Metric :{scores.mean()}")



        torch.cuda.empty_cache()

        gc.collect()

        return np.mean(val_loss_hist)
batch_size = 16
test["text"] = test["text"].str.lower()
test_set = TextDataset(test, tokenizer, is_test=True, return_attention_masks=True)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
for fold, (train_index, valid_index) in enumerate(kf.split(train)):

    print(f"FOLD: {fold}")

    torch.cuda.empty_cache()

    gc.collect()

    print("TRAIN:", train_index, "TEST:", valid_index)

    #X_train, X_test = X[train_index], X[test_index]

    #y_train, y_test = y[train_index], y[test_index]

    train_, val_ = train.loc[train_index], train.loc[valid_index]



    train_set = TextDataset(train_, tokenizer, is_test=False, return_attention_masks=True, max_length=150)

    valid_set = TextDataset(val_, tokenizer, is_test=False, return_attention_masks=True, max_length=150)

    



    train_loader = DataLoader(train_set, batch_size=batch_size)

    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    

'''

    num_epochs = 1



    for epoch in range(num_epochs):

        print(f"\nEpoch: {epoch}")

        model.train()

        #TRAINING LOOP

        

        train_loss = train_fn(model, train_loader, optimizer)

        print(f"Training loss: {train_loss}")

        trn_epoch_loss.append(train_loss)

        

        torch.save(model.state_dict(), "initial_bert_bkp.pth")

        

        print("valid")

        # VALIDATION LOOP

        model.eval()

        val_loss = evaluate(model, valid_loader)

        print(f"valid loss: {val_loss}")

        val_epoch_loss.append(val_loss)



        scheduler.step(val_loss)



'''

#torch.save(model.state_dict(), "/content/drive/My Drive/kaggle/twitter_select_text/initial_bert.pth")
model.eval()

ops_s = []

ops_e = []

with torch.no_grad():

    for i, test_batch in tqdm(enumerate(test_loader), total = len(test_set)//batch_size):

        op = test_batch

        input_ids = op[0].to(device)

        attn_masks = op[1].to(device)

        

        start_p, end_p = model(input_ids, attention_mask=attn_masks)

        ops_s.append(torch.argmax(start_p, 1).cpu())

        ops_e.append(torch.argmax(end_p, 1).cpu())

assert torch.cat(ops_s).shape[0] == test.shape[0]

ops_s = torch.cat(ops_s)

assert torch.cat(ops_e).shape[0] == test.shape[0]

ops_e = torch.cat(ops_e)
for num in range(test.shape[0]):

    test.loc[num, "selected_text"] = test.loc[num, "text"][ops_s[num]:ops_e[num]]

print()
(ops_s>ops_e).sum()
idx = test[test["selected_text"]==""].index
test.loc[idx]
neutral_idx = test[(test["sentiment"]=="neutral") & (test["selected_text"]=="")].index
test.loc[neutral_idx, "selected_text"] = test.loc[neutral_idx, "text"]
non_neutral_idx = test[(test["sentiment"]!="neutral") & (test["selected_text"]=="")].index
test.loc[non_neutral_idx, "selected_text"] = test.loc[non_neutral_idx, "text"]
test.head()
submit = test[["textID", "selected_text"]]
submit
submit.to_csv("submission.csv", index=False)
reopen=pd.read_csv("submission.csv")