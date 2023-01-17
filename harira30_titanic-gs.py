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

import numpy as np

import pandas as pd

import seaborn as sns

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt

from torchvision import transforms

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from torch.utils.data import Dataset, DataLoader, random_split, sampler

from sklearn.metrics import confusion_matrix, classification_report
encoder = LabelEncoder()
dataA = pd.read_csv('/kaggle/input/titanic/train.csv')

dataB = pd.read_csv('/kaggle/input/titanic/test.csv')

dataA.head()
# Check Data Shape

f'Train Data: {dataA.shape} | Test Data: {dataB.shape}'
# Check for Missing Values in Train Data

dataA.isna().sum()
# Check for Missing Values in Test Data

dataB.isna().sum()
# Check if classes are balanced

dataA.groupby('Survived').Survived.count()
# For age we can fill the missing values with mean or median

dataA.Age.mean(), dataA.Age.median()
# Embarked, let find the record for which embarked is missing

dataA[dataA.Embarked.isna()]
# Lets find out if anyone else is in the same cabin

dataA[dataA.Cabin=='B28']
sns.catplot(data=dataA, x='Embarked', kind='count', hue='Pclass')

plt.show()
sns.catplot(data=dataB, x='Embarked', kind='count', hue='Pclass')

plt.show()
# Handle the missing Data

class HandleMissingData():

    def __call__(self, df):

        df.Age.fillna(df.Age.mean(), inplace=True)

        df.Embarked.fillna('S', inplace=True)

        return df
dataA['Pclass'].unique()
# Lets Encode Categorical value in our dataset.

class EncodeCat():

  def __init__(self):

    self.lEncoder = LabelEncoder()

    self.oEncoder= OneHotEncoder(sparse=False)

  def __call__(self, df):

    embarked_ie = self.lEncoder.fit_transform(df.Embarked.astype('str'))

    embarked_ie = embarked_ie.reshape(len(embarked_ie), 1)

    df[['C', 'Q', 'S']] = pd.DataFrame(self.oEncoder.fit_transform(embarked_ie),index=df.index)

    gender_ie = self.lEncoder.fit_transform(df.Sex.astype('str'))

    gender_ie = gender_ie.reshape(len(gender_ie), 1)

    df[['F','M']] = pd.DataFrame(self.oEncoder.fit_transform(gender_ie))

    pclass = df['Pclass'].values

    pclass = pclass.reshape(len(pclass), 1)

    df[['c1','c2', 'c3']] = pd.DataFrame(self.oEncoder.fit_transform(pclass))

    return df
# Lets Explore SibSp,Parch

sns.catplot(data=dataA, x='SibSp', kind='count')

sns.catplot(data=dataA, x='SibSp', y='Survived', kind='bar')

sns.catplot(data=dataA, x='Parch', kind='count')

sns.catplot(data=dataA, x='Parch', y='Survived', kind='bar')
# Lets convert Age to buckets

class AgeGroup():

  def __call__(self, df):

    binss = [0,9.9,19.9,29.9,39.9,49.9,59.9,69.9,75.9,df.Age.max()+1]

    df['agec'] = pd.cut(df['Age'],bins=binss, labels=False)

    return df
# Lets transform our data

transform_data = transforms.Compose([HandleMissingData(), EncodeCat(), AgeGroup()])

TdataA = transform_data(dataA)

TdataB = transform_data(dataB)
# Lets correlate our data 

dataACor = TdataA.corr()

plt.figure(figsize=(14,6))

sns.heatmap(dataACor, xticklabels=dataACor.columns, yticklabels=dataACor.columns, annot=True)
# lets balance the dataset

TdataA = TdataA.groupby('Survived').apply(lambda x: x.sample(n=342)).reset_index(drop=True)

target_counts = TdataA.groupby('Survived').Survived.count()

target_counts
# Lets select Features

features = ['PassengerId', 'c1','c2','c3','F','M','C','Q','S','agec','SibSp', 'Parch']

train_features = TdataA[features]

test_features = TdataB[features]

target = TdataA['Survived']
class ToTensor():



  def __call__(self, sample):

    x = torch.from_numpy(sample[0].astype(float)).type(torch.FloatTensor)

    y = torch.tensor(sample[1], dtype=torch.float64)

    pid = sample[2]

    return x,y, pid
class TitanicDataSet(Dataset):



  def __init__(self, features, dependent=None, transform=None):

    self.features = features

    self.target = dependent

    self.transform = transform

  

  def __len__(self):

    return len(self.features)

  

  def __getitem__(self, idx):

    t = self.target[idx] if self.target is not None else -1.0

    sample = self.features[idx][1:], t, self.features[idx][0]

    if self.transform:

      sample = self.transform(sample)

    return sample
data_trans = transforms.Compose([ToTensor()])
titanic_data =  TitanicDataSet(train_features.values, target, data_trans)
records = len(train_features)

train_len = int(0.8*records)

val_len = records - train_len

print(f'Total: {records}, Train: {train_len}, Val: {val_len}')

train_data, val_data = random_split(titanic_data, [train_len, val_len])

train_do = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

val_do = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=4)
class TPM(nn.Module):



  def __init__(self):

    super(TPM, self).__init__()

    self.sequence = nn.Sequential(

        nn.Linear(11, 22),

        nn.ReLU(),

        nn.Linear(22, 11),

        nn.ReLU(),

        nn.Linear(11,1),

        nn.Sigmoid()

    )

  

  def forward(self, x):

    return self.sequence(x)
net = TPM()

net
def compute_accuracy(yhat, y):

  y_pred_tag = torch.round(yhat)

  correct_results_sum = (y_pred_tag == y).sum().float()

  acc = correct_results_sum/y.shape[0]

  acc = torch.round(acc * 100)

  return acc
EPOCHS = 100

LR = 0.0005

ls_func = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(net.parameters(), lr=LR)
net.train()

lloss, accu = [],[]

for e in range(EPOCHS):

  epoch_loss = 0

  epoch_acc = 0

  for X, y, pid in train_do:

    optimizer.zero_grad()

    yhat = net(X)

    loss = ls_func(yhat, y.unsqueeze(1))

    acc = compute_accuracy(yhat, y.unsqueeze(1))

    loss.backward()

    optimizer.step()

    

    epoch_loss+=loss.item()

    epoch_acc+=acc.item()

  elos = epoch_loss/len(train_do)

  eacc = epoch_acc/len(train_do)

  lloss.append(elos)

  accu.append(eacc)

  print(f'Epoch {e+0:03}: | Loss: {elos:.5f} | Acc: {eacc:.3f}')



plt.figure(figsize=(16,4))

xval = np.arange(0, len(lloss), 1)

for i, m in enumerate([lloss, accu]):

  metric = ['Loss', 'Accuracy']

  color = ['r', 'g']

  plt.subplot(1,2,i+1)

  plt.plot(xval, m, c=color[i])

  plt.title(f'{metric[i]} Prop')

  plt.xlabel('Epochs')

  plt.ylabel(f'{metric[i]}')

plt.show()
net.eval()

correct, yhats, actual_y = 0, [], []

failed_features = []

val_do_len = len(val_do)

with torch.no_grad():

  for X, y, pid in val_do:

    actual_y.append(y)

    pred = net(X)

    yhat = torch.round(pred)

    yhats.append(yhat.cpu().numpy())

    if yhat == y:

      correct+=1

    else:

      row_rec=(X.numpy()[0])

      row_rec = np.append(row_rec, yhat.unsqueeze(1).tolist())

      row_rec = np.insert(row_rec, 0, pid)

      failed_features.append(row_rec)

ypreds = [a.squeeze().tolist() for a in yhats]

ac = (correct / val_do_len)*100



print(f'Test Data: {val_do_len} :: Wrong:{val_do_len - correct} :: Correct:{correct} :: Acc: {ac}')

wc_features = pd.DataFrame(failed_features,columns=features+['Pred'])
sns.heatmap(confusion_matrix(ypreds, actual_y), annot=True, fmt='d')
tesd = TitanicDataSet(features=test_features.values, transform=data_trans)

test_do = DataLoader(tesd, batch_size=1, shuffle=False, num_workers=4)
result = pd.DataFrame(columns=['PassengerId', 'Survived'])

passe, sur = [], []

with torch.no_grad():

  for X, y, pid in test_do:

    pred = net(X)

    yhat = torch.round(pred)

    passe.append(pid.item())

    sur.append(int(yhat.item()))



result['PassengerId'] = passe

result['Survived'] = sur



result.to_csv(f'/kaggle/working/titanic_bc_result_{ac}.csv')

result.groupby('Survived', as_index=False).count()
ar = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

ar.groupby('Survived', as_index=False).count()
def compute_val_accuracy(yhat, y):

  correct_results_sum = (yhat == y).sum()

  acc = correct_results_sum/len(y)

  acc = (acc * 100)

  return acc
compute_val_accuracy(result.Survived, ar.Survived)