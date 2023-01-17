# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import transformers

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.model_selection import train_test_split
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.head()
X = train_df.loc[:,'text']

y = train_df.loc[:,'target']
max_len = 0

for text in X:

    max_len = max(max_len, len(text))

max_len 
class Dataset(torch.utils.data.Dataset):

    def __init__(self,df,y=None,max_len=164):

        self.df = df

        self.y = y

        self.max_len= max_len

        self.tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')

        

    def __getitem__(self,index):

        row = self.df.iloc[index]

        ids,masks = self.get_input_data(row)

        data = {}

        data['ids'] = ids

        data['masks'] = masks

        if self.y is not None:

            data['out'] = torch.tensor(self.y.iloc[index],dtype=torch.float32)

        return data

    

    def __len__(self):

        return len(self.df)

    

    def get_input_data(self,row):

        row = self.tokenizer.encode(row,add_special_tokens=True,add_prefix_space=True)

        padded = row + [0]* (max_len - len(row))

        padded = torch.tensor(padded,dtype=torch.int64)

        mask = torch.where(padded != 0 , torch.tensor(1),torch.tensor(0))

        return padded,mask

        

    

    
train_x,val_x,train_y,val_y = train_test_split(X,y,test_size=0.2,stratify=y)

train_loader = torch.utils.data.DataLoader(Dataset(train_x,train_y),batch_size=16,shuffle=True,num_workers=2)

val_loader = torch.utils.data.DataLoader(Dataset(val_x,val_y),batch_size=16,shuffle=False,num_workers=2)
next(iter(train_loader))
class Model(nn.Module):

    def __init__(self):

        super(Model,self).__init__()

        self.distilBert = transformers.RobertaModel.from_pretrained('roberta-base')

        self.l0 = nn.Linear(768,512)

        self.l1 = nn.Linear(512,256)

        self.l2 = nn.Linear(256,1)

        self.d0 = nn.Dropout(0.5)

        self.d1 = nn.Dropout(0.5)

        self.d2 = nn.Dropout(0.5)

        nn.init.normal_(self.l0.weight,std=0.2)

        nn.init.normal_(self.l1.weight,std=0.2)

        nn.init.normal_(self.l2.weight,std=0.2)

    

    def forward(self,ids,masks):

        hid = self.distilBert(ids,attention_mask=masks)

        hid = hid[0][:,0]

        x = self.d0(hid)

        x = self.l0(x)

        x = F.leaky_relu(x)

        x = self.d1(x)

        x = self.l1(x)

        x = F.leaky_relu(x)

        x = self.d2(x)

        x = self.l2(x)

        return x

    

    
model = Model().to('cuda')

criterion = nn.BCEWithLogitsLoss(reduction='mean')

optimizer = torch.optim.AdamW(model.parameters(),lr=3e-5)
def accuracy_score(outputs,labels):

    outputs = torch.round(torch.sigmoid(outputs))

    correct = (outputs == labels).sum().float()

    return correct/labels.size(0)
from tqdm import tqdm
epochs = 4

for epoch in range(epochs):

    epoch_loss = 0.

    model.train()

    for data in tqdm(train_loader):

        ids = data['ids'].cuda()

        masks = data['masks'].cuda()

        labels = data['out'].cuda()

        labels = labels.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(ids,masks)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch{epoch+1} loss : {epoch_loss/len(train_loader)}")

    print(f"Epoch{epoch+1} accuracy : ",accuracy_score(outputs,labels).item())

    val_loss = 0.

    model.eval()

    for data in tqdm(val_loader):

        ids = data['ids'].cuda()

        masks = data['masks'].cuda()

        labels = data['out'].cuda()

        labels = labels.unsqueeze(1)

        outputs = model(ids,masks)

        loss = criterion(outputs,labels)

        val_loss += loss.item()

    

    print(f"Epoch{epoch+1} val loss : {val_loss/len(val_loader)}")

    print(f"Epoch{epoch+1} val accuracy : ",accuracy_score(outputs,labels).item())

        

        

        
test_loader = torch.utils.data.DataLoader(Dataset(test_df['text'],y=None),batch_size=16,shuffle=False,num_workers=2)
preds = []

for data in test_loader:

    ids = data['ids'].cuda()

    masks = data['masks'].cuda()

    model.eval()

    outputs = model(ids,masks)

    preds += outputs.cpu().detach().numpy().tolist()

pred = np.round(1/(1 + np.exp(-np.array(preds))))
pred = np.array(pred,dtype=np.uint8)
sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sub['target'] = pred
sub.to_csv('submission.csv',index=False)
sub.head(30)