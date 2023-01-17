import numpy as np

import pandas as pd

import torch

import transformers

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')
import torch.nn as nn

import torch.nn.functional as F

from sklearn.metrics import accuracy_score

import torch
class Dataset(torch.utils.data.Dataset):

    def __init__(self,df,out,max_len=96):

        

        self.df = df

        self.out = out

        self.max_len = max_len

        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        

    def __getitem__(self,index):

        data = {}

        row = self.df.iloc[index]

        ids,masks,labels = self.get_input_data(row)

        data['ids'] = torch.tensor(ids)

        data['masks'] = masks

        data['labels'] = torch.tensor(self.out.iloc[index].values[0],dtype=torch.float32)

        return data

    

    def __len__(self):

        return len(self.df)

    

    def get_input_data(self,row):

        ids = self.tokenizer.encode(row[0],add_special_tokens=True)

        pad_len = self.max_len - len(ids)

        if pad_len > 0 :

            ids += [0]*pad_len

        ids = torch.tensor(ids)    

        masks = torch.where(ids != 0 , torch.tensor(1),torch.tensor(0))

        return ids,masks,self.out.iloc[0].values

    

        
df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv',delimiter='\t',header=None)

train_set,val_set = train_test_split(df,test_size = 0.2)

train_loader = torch.utils.data.DataLoader(Dataset(pd.DataFrame(train_set[0]),pd.DataFrame(train_set[1]),max_len=80),batch_size = 32, shuffle = True, num_workers = 2)

val_loader = torch.utils.data.DataLoader(Dataset(pd.DataFrame(val_set[0]),pd.DataFrame(val_set[1]),max_len=80),batch_size = 32,num_workers = 2)
def binary_acc(y_pred, y_test):

    y_pred_tag = torch.round(torch.sigmoid(y_pred))



    correct_results_sum = (y_pred_tag == y_test).sum().float()

    acc = correct_results_sum/y_test.shape[0]

    acc = torch.round(acc * 100)

    

    return acc.item()
class Model(nn.Module):

    def __init__(self):

        super(Model,self).__init__()

        config = transformers.DistilBertConfig.from_pretrained('distilbert-base-uncased')

        self.distilBert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased',config=config)

        self.fc0 = nn.Linear(768,512)

        self.d0 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512,256)

        self.d1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256,1)

        nn.init.normal_(self.fc0.weight,std= 0.1)

        nn.init.normal_(self.fc0.bias ,0.)

        nn.init.normal_(self.fc1.weight,std =0.1)

        nn.init.normal_(self.fc1.bias, 0.)

        nn.init.normal_(self.fc2.weight,std=0.1)

        nn.init.normal_(self.fc2.bias , 0.)



        

    def forward(self,input_ids,attention_mask):

        hid= self.distilBert(input_ids,attention_mask = attention_mask)

        hid= hid[0][:,0]

        x = self.fc0(hid)

        x = F.relu(x)

        x = self.d0(x)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.d1(x)

        return self.fc2(x)

criterion = nn.BCEWithLogitsLoss(reduction='mean').to('cuda')

model = Model().to('cuda')

for params in model.distilBert.parameters():

    params.require_grad = False

    params._trainable = False

optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)


for epoch in range(3):

        

        epoch_loss = 0

        val_loss = 0

        correct = 0

        accuracy = 0



        for data in train_loader:

            ids = data['ids'].cuda()

            masks = data['masks'].cuda()

            labels = data['labels'].cuda()

            labels = labels.unsqueeze(1)

            model.train()

            optimizer.zero_grad()

            outputs = model(ids,masks)

            loss = criterion(outputs,labels)

            loss.backward()

            optimizer.step()

           

            epoch_loss += loss.item()

            

        print(f'Train Epoch {epoch} : Loss {epoch_loss/len(train_loader)}')

        print("Train Accuracy : ",binary_acc(outputs,labels))

        model.eval()

        correct = 0

        for data in val_loader:

            ids = data['ids'].cuda()

            masks = data['masks'].cuda()

            labels = data['labels'].cuda()

            outputs = model(ids,masks)

            labels = labels.unsqueeze(1)

            loss = criterion(outputs,labels)

            val_loss += loss.item()

            

        print(f'Val Epoch {epoch} : Loss {val_loss/len(val_loader)}')

        print("Val Accuracy : ",binary_acc(outputs,labels))

        

def test(sentence):

    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    ids = tokenizer.encode(sentence,add_special_tokens=True)

    padded = ids + [0]*(80 - len(ids))

    padded = torch.tensor(padded,dtype=torch.int64).unsqueeze(0)

    masks = torch.where(padded != 0 , torch.tensor(1), torch.tensor(0)).cuda()

    padded = padded.cuda()

    model.eval()

    output = model(padded,masks)

    return torch.round(F.sigmoid(output)).item()
test("Kaggle is a very nice platform to start Deep Learning.")