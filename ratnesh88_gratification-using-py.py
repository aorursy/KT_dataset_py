import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, torch

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import confusion_matrix, classification_report

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train.head()
device = torch.device('cpu')
cols = train.columns

cols
predictor = cols[1:-1]

target = cols[-1]
## Visualixation/Stat

train[predictor].describe()
sns.countplot(train['target']);
## Splits/Transform

X = torch.Tensor(train[predictor].values)

Y = torch.Tensor(train['target'].values)
X.shape[0]
## Model

class Model(torch.nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.l1 = torch.nn.ReLU(256)

        self.l2 = torch.nn.Linear(256,1)

    

    def forward(self,x):

        f1 = self.l1(x)

        pred = self.l2(f1)

        return pred



model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

criterion = torch.nn.MSELoss()
## Train

for i in range(10):

    print(f'Epoch-{i}')

    for j in range(0,X.shape[0],256):

        y_pred = model(X[j:j+256])

        losses = criterion(y_pred, Y[j:j+256])



        optimizer.zero_grad()

        losses.backward()

        optimizer.step()

    print(f"Loss-{losses.item()}")
## Prediction

preds = model(X).flatten() > 0.465

preds = preds.tolist()

sns.countplot(preds);
## Evaluation

c = confusion_matrix(train['target'].values, preds)

sns.heatmap(c, annot=True)
## Test 

test = pd.read_csv('../input/test.csv')

print(test.shape)

test.head()
test_pred = model(torch.Tensor(test[predictor].values))

test_pred = test_pred.flatten().tolist()
## Submission

submission = pd.read_csv('../input/sample_submission.csv')

submission.head()
submission['target'] = test_pred
submission.to_csv('output.csv',index=False)