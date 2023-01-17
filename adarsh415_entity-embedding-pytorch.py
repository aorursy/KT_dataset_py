# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import torch

import torch.nn as nn

from sklearn import preprocessing

from tqdm import tqdm
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

sample = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')
#https://www.kaggle.com/pchlq82/simple-entity-embeddings-with-pytorch
train.shape, test.shape
test['target'] = -1
data = pd.concat([train, test], axis=0).reset_index(drop=True)
data.shape
features = [f for f in train.columns if f not in ['target', 'id']]
for f in features:

    

    lbl_enc = preprocessing.LabelEncoder()

    data.loc[:,f] = lbl_enc.fit_transform(data[f].astype(str).fillna('-1').values)
class EntitySet:

    def __init__(self, data, target=None):

        self.data = data.astype(np.int64)

        self.n = len(self.data)

        

        if target is None:

            self.target = np.zeros((self.n))

        else:

            self.target = target.astype(np.float32)

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, item):

        

        return [self.data[item],self.target[item]]
train = data[data.target != -1]

test = data[data.target == -1].drop(columns=['target'], axis=1)
train.shape, test.shape
train_dataloader = torch.utils.data.DataLoader(EntitySet(train[features].values, train.target.values), batch_size=64)

test_dataloader = torch.utils.data.DataLoader(EntitySet(test[features].values), batch_size=128)
# for b, (data, target) in enumerate(train_dataloader):

#     print(b)

#     print(data[:,1])
class EntityModel(nn.Module):

    def __init__(self,df, features):

        super(EntityModel, self).__init__()

        self.df = df

        self.features = features

        self.embed = self._build_embedding(self.df)

        no_of_embed = sum([v for v,_,_ in self.embed])

        self.linear_1 = nn.Linear(no_of_embed, 300)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.linear_2 = nn.Linear(300, 1)

        self.sigmoid = nn.Sigmoid()

        

        

    @staticmethod

    def _build_embedding(df):

        embed_list = []

        feat_index = []

        for f in features:

            embed_size = int(df[f].nunique())

            embed_dim = int(min(np.ceil(embed_size / 2), 50))

            ids = {val:i for i, val in enumerate(set(df[f].values))}

            embed = nn.Embedding(embed_size+2000, embed_dim)

            embed_list.append((embed_dim, embed, ids))

        return embed_list

        

    def forward(self, xb):

        batch_size, feat = xb.shape[0], xb.shape[1]

        out_list = []

        for i in range(feat):

            data = xb[:,i]

            xe = self.embed[i][1](data)

            xe = xe.view(-1, self.embed[i][0])

            out_list.append(xe)

        cancated = torch.cat(out_list, 1)

        l1 = self.linear_1(cancated)

        l2 = self.relu(l1)

        l3 = self.dropout(l2)

        out = self.linear_2(l3)

        out = self.sigmoid(out)

        return out

    
model = EntityModel(train, features)


def train_fn(model, train_dataloader, epochs=10):

    model.train()   

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = nn.BCELoss()

    for epoch in range(epochs):

        epoch_loss = 0

        for b, (data, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader) ):

            out = model(data)



            #compute loss

            loss = loss_fn(out, target)

            

            epoch_loss = loss



            # make sure grad is zero

            optimizer.zero_grad()



            #backward pass

            loss.backward()



            # Calling the step function on an Optimizer makes an update to its

            # parameters

            optimizer.step()

        if epoch%5==0:

            print(epoch, epoch_loss.item())
train_fn(model, train_dataloader, epochs=20)
def predict(model, data_loader):

    model.eval()

    result_append = []

    result_extend = []

    with torch.no_grad():

        for b, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader) ):

            out = model(data)

            to_out = out.detach().cpu().numpy()

            result_append.append(to_out)

            result_extend.extend(to_out.tolist())

    return result_append, result_extend

        
result_append, result_extend = predict(model, test_dataloader)
test= np.concatenate(result_append, axis=0)
sample['target']= test
sample.head()
sample.to_csv('entity_embedding_torch.csv', index=False)