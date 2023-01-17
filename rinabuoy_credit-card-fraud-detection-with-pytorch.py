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
import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

plt.hist(data['Class'], color='red')

plt.xlabel('Class')

plt.ylabel('Transaction')

plt.title('Class Imbalance', fontsize=15)
pc_fraud = len(data.loc[data['Class'] == 1].values)/len(data.loc[data['Class'] == 0].values)

print(pc_fraud*100)

print(data.head())
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from collections import Counter



# Original dataset

x = data.drop('Class', axis=1).values

y = data['Class'].values

scaler = MinMaxScaler()

scaler.fit(x)

x = scaler.transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)



print('Sampled train dataset shape %s' % Counter(ytrain))

print('Sampled validation dataset shape %s' % Counter(ytest))
import torch



bs =100



#creating torch dataset and loader using original dataset. 

#to use resampled dataset, replace ex. xtrain with xtrain_over etc.

train_ds = torch.utils.data.TensorDataset(torch.tensor(xtrain).float(), torch.tensor(ytrain).float())

valid_ds = torch.utils.data.TensorDataset(torch.tensor(xtest).float(), torch.tensor(ytest).float())



train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs)

valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=bs)
#network class 2-hidden layer model

class Classifier(torch.nn.Module):

    def __init__(self, n_input=10, n_hidden = 20, n_output = 1,drop_prob=0.5):

        super().__init__()

        self.extractor1 = torch.nn.Linear(n_input, n_hidden)

        self.extractor2 = torch.nn.Linear(n_hidden, n_hidden)

        self.relu = torch.nn.ReLU()

        self.drop_out = torch.nn.Dropout(drop_prob)

        self.classifier = torch.nn.Linear(n_hidden, n_output)



    def forward(self, xb):

        x = self.relu(self.extractor1(xb))

        x = self.relu(self.extractor2(x))

        x = self.drop_out(x)

        return self.classifier(x).squeeze()

def loss_batch(model, loss_func, xb, yb, opt=None):

    loss = loss_func(model(xb), yb)



    if opt is not None:

        loss.backward()

        opt.step()

        opt.zero_grad()



    return loss.item(), len(xb)
#training the network

def train(epochs, model, loss_func, opt, train_dl, valid_dl):

    for epoch in range(epochs):

        model.train()

        for xb, yb in train_dl:

            loss_batch(model, loss_func, xb, yb, opt)



        model.eval()

        with torch.no_grad():

            losses, nums = zip(

                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]

            )

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)



        print(epoch, val_loss)

#network setting

n_input = xtrain.shape[1]

n_output = 1

n_hidden = 15



model = Classifier(n_input=n_input,n_hidden=n_hidden,n_output=n_output,drop_prob=0.2)



lr = 0.001



#for orignal dataset, I use pos_weight.

pos_weight = torch.tensor([5])

opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)



n_epoch = 200
train(n_epoch,model,loss_func,opt,train_dl,valid_dl)

model.eval()

from sklearn import metrics

ypred = model(torch.tensor(xtest).float()).detach().numpy()



ypred [ypred>=0.5] =1.0

ypred [ypred<0.5] =0.0

print('Confusion matrix: {}'. format(metrics.confusion_matrix(ytest, ypred)))

print('AUPRC score: {}'. format(metrics.average_precision_score(ytest, ypred)))

print('AUROC score: {}'.format(metrics.roc_auc_score(ytest, ypred)))

print('Accuracy score: {}'.format(metrics.accuracy_score(ytest, ypred)))

print(metrics.classification_report(ytest, ypred))