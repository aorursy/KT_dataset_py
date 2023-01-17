# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) 

# will list the files in the input directory



import os

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
(train.shape, test.shape)
all_df = pd.concat([train, test], sort=False)
all_df.head()
def preprocess(df, cat_cols):

  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

  

  for cat_col in cat_cols:

    if cat_col in ['Embarked']:

      df[cat_col] = LabelEncoder().fit_transform(df[cat_col].astype(str))

    else:

      df[cat_col] = LabelEncoder().fit_transform(df[cat_col])

  

  df = df.fillna(df.mean())

  return df
cat_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

all_df = preprocess(all_df, cat_cols)

all_df.head()
train_df = all_df.head(train.shape[0])

train_df.shape
class TabularDataset(Dataset):

  def __init__(self, df, categorical_columns, output_column=None):

    super().__init__()

    self.len = df.shape[0]

    

    self.categorical_columns = categorical_columns

    self.continous_columns = [col for col in df.columns if col not in self.categorical_columns + [output_column]]

    

    if self.continous_columns:

      self.cont_X = df[self.continous_columns].astype(np.float32).values

    else:

      self.cont_X = np.zeros((self.len, 1))

      

    if self.categorical_columns:

      self.cat_X = df[self.categorical_columns].astype(np.int64).values

    else:

      self.cat_X = np.zeros((self.len, 1))

      

    if output_column != None:

      self.has_label = True

      self.label = df[output_column].astype(np.float32).values.reshape(-1, 1)

    else:

      self.has_label = False

  

  def __len__(self):

    return self.len

  

  def __getitem__(self, index):

    if self.has_label:

      return [self.label[index], self.cont_X[index], self.cat_X[index]]

    else:

      return [self.cont_X[index], self.cat_X[index]]
train_ds = TabularDataset(train_df, cat_cols, 'Survived')

train_dl = DataLoader(train_ds, 64, shuffle=True)
len(train_ds)
class TitanicNet(nn.Module):

  def __init__(self, emb_dims, n_cont, lin_layer_sizes, output_size):

    super().__init__()

    self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])



    self.n_embs = sum([y for x, y in emb_dims])

    self.n_cont = n_cont



    # Linear Layers

    first_lin_layer = nn.Linear(self.n_embs + self.n_cont, lin_layer_sizes[0])



    self.lin_layers = nn.ModuleList(

        [first_lin_layer] + 

        [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)]

    )

    

#     for lin_layer in self.lin_layers:

#       nn.init.kaiming_normal_(lin_layer.weight.data)



    # Output Layer

    self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)

    nn.init.kaiming_normal_(self.output_layer.weight.data)



    # Batch Norm Layers

    self.first_bn_layer = nn.BatchNorm1d(self.n_cont)

    self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

    

  def forward(self, cont_data, cat_data):

    if self.n_embs != 0:

      x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]

      x = torch.cat(x, 1)

      

    if self.n_cont != 0:

      normalized_cont_data = self.first_bn_layer(cont_data)



      if self.n_embs != 0:

        x = torch.cat([x, normalized_cont_data], 1) 

      else:

        x = cont_data

        

    for lin_layer, bn_layer in zip(self.lin_layers, self.bn_layers):

      x = torch.relu(lin_layer(x))

      x = bn_layer(x)



    x = self.output_layer(x)

    x = torch.sigmoid(x)

    return x
cat_dims = [int(all_df[col].nunique()) for col in cat_cols]

cat_dims
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

emb_dims
torch.manual_seed(2)
model = TitanicNet(emb_dims, n_cont=2, lin_layer_sizes=[50, 100, 50], output_size=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

no_of_epochs = 10

criterion = nn.BCELoss()



for epoch in range(no_of_epochs):

  epoch_loss = 0

  epoch_accuracy = 0

  i = 0

  for y, cont_x, cat_x in train_dl:

    preds = model(cont_x, cat_x)

    loss = criterion(preds, y)

    epoch_loss += loss

    

    accuracy = ((preds > 0.5).float() == y).float().mean()

    epoch_accuracy += accuracy



    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  

  print("Epoch ", epoch, ", loss: ", epoch_loss.item()/len(train_dl), "accuracy: ", epoch_accuracy.item()/len(train_dl))

  
test_df = all_df.tail(test.shape[0])

test_ds = TabularDataset(test_df, cat_cols, 'Survived') # The label is actually useless. But to keep our code consistent, we leave it here.

test_dl = DataLoader(test_ds, len(test_ds))
with torch.no_grad():

  for _, cont_x, cat_x in test_dl:

    preds = model(cont_x, cat_x)

    preds = (preds > 0.5)
preds.flatten().shape
output_df = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':preds.flatten().numpy()})
output_df.head()
output_df.to_csv('titanic_preds.csv', index=False)