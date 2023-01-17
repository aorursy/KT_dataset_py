from sklearn.datasets import fetch_20newsgroups

import random

from transformers import AutoModel, BertTokenizerFast, AdamW

import re

import pandas as pd

import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn

import numpy as np

from sklearn.metrics import classification_report
#get data

categories = ['alt.atheism', 'comp.graphics', 'rec.autos', 'sci.space']

train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
print("Type of dataset", type(train_data))

print("Keys", train_data.keys())
#this cell show a random text for a defined category

picked_cat = 'alt.atheism'

print("RANDOM TEXT for {}".format(picked_cat))

print(train_data['data'][np.random.choice(np.where(train_data['target']==train_data['target_names'].index(picked_cat))[0])])
print("Train data shape", len(train_data['data']))

print("Test data shape", len(test_data['data']))
#import bert model and tokenizer

bert = AutoModel.from_pretrained('bert-base-uncased')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
sample_text = [np.random.choice(train_data['data'])[:100] for _ in range(2)]

print(sample_text)

print(tokenizer.batch_encode_plus(sample_text, padding=True, return_token_type_ids=False))
text_data = train_data['data'] + test_data['data']

print(len(text_data))
def preprocess(text):

    space_re = re.compile("\s+")

    sp_chr_re =  re.compile("([^\w\d\s]|_)+")

    cleaned_text = space_re.sub(" ", text)

    while True:

        temp_text = sp_chr_re.sub(r"\1", cleaned_text)

        if temp_text==cleaned_text:

            break

        cleaned_text = temp_text

    return cleaned_text
#lets do some cleaning

random_sample_text = np.random.choice(train_data['data'])

print(random_sample_text)

print("==================CLEANED===================")

print(preprocess(random_sample_text))
#define max text length

len_seq = pd.Series([len(text.split()) for text in train_data['data']])

len_seq[len_seq<1000].hist(bins=30)
MAX_LEN = 500
tokens_train = tokenizer.batch_encode_plus([preprocess(text) for text in train_data['data']], 

                                           max_length=MAX_LEN, 

                                           pad_to_max_length=True, 

                                           truncation=True)

tokens_test = tokenizer.batch_encode_plus([preprocess(text) for text in test_data['data']], 

                                           max_length=MAX_LEN, 

                                           pad_to_max_length=True, 

                                           truncation=True)
#UPDATE: pytorch crossentropyloss does not take one hot encoded vector rather a 1-D array with class indices

#one hot encode train and test labels 

train_labels = np.zeros((train_data['target'].size, train_data['target'].max()+1))

train_labels[np.arange(train_data['target'].size),train_data['target']]=1



test_labels = np.zeros((test_data['target'].size, test_data['target'].max()+1))

test_labels[np.arange(test_data['target'].size),test_data['target']]=1
#convert all to tensors

train_seq = torch.tensor(tokens_train['input_ids'])

train_mask = torch.tensor(tokens_train['attention_mask'])

train_y = torch.tensor(train_data['target'], dtype=torch.long)



test_seq = torch.tensor(tokens_test['input_ids'])

test_mask = torch.tensor(tokens_test['attention_mask'])

test_y = torch.tensor(test_data['target'], dtype=torch.long)
#creat pytorch dataloaders to load data into model in batches

BATCH_SIZE = 32



train_tensor_data = TensorDataset(train_seq, train_mask, train_y)

train_sampler = RandomSampler(train_tensor_data)

train_data_loader = DataLoader(train_tensor_data, sampler=train_sampler, batch_size=BATCH_SIZE)



test_tensor_data = TensorDataset(test_seq, test_mask, test_y)

test_sampler = RandomSampler(test_tensor_data)

test_data_loader = DataLoader(test_tensor_data, sampler=test_sampler, batch_size=BATCH_SIZE)
#freeze parameters of BERT

for param in bert.parameters():

    param.requires_grad=False
class BERT_Model(nn.Module):

    def __init__(self, bert):

        super(BERT_Model, self).__init__()

        self.bert=bert

        #dropout layer

        self.dropout = nn.Dropout(0.1)

        #relu activation fn

        self.relu = nn.ReLU()

        #dense layer 1

        self.fc1 = nn.Linear(768,512)

        #dense layer 2

        self.fc2 = nn.Linear(512,4)

        #softmax

        self.softmax = nn.Softmax(dim=1)

    

    def forward(self, sent_id, mask):

        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.softmax(x)

        return x
device = torch.device('cuda')
model = BERT_Model(bert)

#push model to GPU

model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
#define loss

cross_entropy  = nn.CrossEntropyLoss()
#training epochs

EPOCHS = 10
def train():

    model.train()

    total_loss, total_accuracy = 0, 0

    total_preds = []

    for step, batch in enumerate(train_data_loader):

        #progress update after every 50 batches

        if step%50==0 and not step==0:

            print("Batch {} of {}".format(step, len(train_data_loader)))

        #push batch to gpu

        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        model.zero_grad()

        #get model predictions with current batch

        preds = model(sent_id, mask)

        #compute loss between actual and predicted values

        loss = cross_entropy(preds, labels)

        total_loss = total_loss + loss.item()

        #backward pass to calculate gradients

        loss.backward()

        #clip gradients to 1, prevents exploding gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #update params

        optimizer.step()

        #push preds to CPU

        preds = preds.detach().cpu().numpy()

        #append the model, predictions

        total_preds.append(preds)

    #compute loss

    avg_loss = total_loss/len(train_data_loader)

    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds
#function to evaluate the model

def evaluate():

    print("\nEvaluating....")

    #deactivate dropout layers

    model.eval()

    total_loss, total_accuracy = 0,0

    #empty list to save model prediction

    total_preds = []

    #iterate over batches

    for step, batch in enumerate(test_data_loader):

        #progress report every 50 epochs

        if step%50==0 and not step==0:

            print("Batch {} of {}".format(step, len(test_data_loader)))

        #push batch to GPU

        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        #deactivate autograd

        with torch.no_grad():

            #model predictions

            preds = model(sent_id, mask)

            #compute the test loss 

            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    #compute val loss for epoch

    avg_loss = total_loss/len(test_data_loader)

    #concatenate

    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds
#set loss to inf

best_test_loss = float('inf')



#training and val loss list

train_losses = []

val_losses = []



#loop over epochs

for epoch in range(EPOCHS):

    print("Epoch {}/{}".format(epoch+1, EPOCHS))

    #train_model

    train_loss, _ = train()

    #evaluate model

    val_loss, _ = evaluate()

    #save the best model

    if val_loss<best_test_loss:

        best_test_loss = val_loss

        torch.save(model.state_dict(), 'saved_weights.pt')

    #append training and val loss

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    print("Training Loss: ", train_loss)

    print("Test Loss: ", val_loss)
#load weights of best model

path = 'saved_weights.pt'

model.load_state_dict(torch.load(path))
test_preds = []

for i in range(len(test_seq)):

    test_preds.append(model(test_seq[i][None, ...].to(device), test_mask[i][None, ...].to(device)))
for idx, val in enumerate(test_preds):

    test_preds[idx] = val.detach().cpu().numpy()
test_preds = np.array(test_preds).squeeze()
final_val_preds = np.argmax(test_preds, axis=1)
print(classification_report(test_data['target'], final_val_preds))