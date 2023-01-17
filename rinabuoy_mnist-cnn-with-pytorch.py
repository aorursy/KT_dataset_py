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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn import metrics

from imblearn.over_sampling import SMOTE 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from collections import Counter



import torch
train_on_gpu = torch.cuda.is_available()

if(train_on_gpu):

    print('Training on GPU!')

else: 

    print('No GPU available, training on CPU; consider making n_epochs very small.')
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

plt.hist(data['label'], color='red')

plt.xlabel('Label')

plt.ylabel('Count')

plt.title('Class distribution', fontsize=15)
x = data.drop('label', axis=1).values

y = data['label'].values

x= x/ 255.

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2, random_state=0)



print('Sampled train dataset shape %s' % Counter(ytrain))

print('Sampled validation dataset shape %s' % Counter(yvalid))
bs =16



#creating torch dataset and loader using original dataset. 

#to use resampled dataset, replace ex. xtrain with xtrain_over etc.

train_ds = torch.utils.data.TensorDataset(torch.tensor(xtrain).float(), torch.tensor(ytrain).long())

valid_ds = torch.utils.data.TensorDataset(torch.tensor(xvalid).float(), torch.tensor(yvalid).long())



train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs)

valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=bs)
#network class 2-hidden layer model

class Classifier(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        #self.conv3 = torch.nn.Conv2d(16, 10, kernel_size=3, stride=1, padding=1)

        self.relu = torch.nn.ReLU()

        self.max_pool2d = torch.nn.MaxPool2d(2)

        self.dropout = torch.nn.Dropout(0.5)

        self.clf1 = torch.nn.Linear(576,128)

        self.clf2 = torch.nn.Linear(128,10)







    def forward(self, xb):

        xb = xb.view(-1, 1, 28, 28)

        bs = xb.shape[0]

        xb = self.max_pool2d(self.relu(self.conv1(xb)))

        xb = self.max_pool2d(self.relu(self.conv2(xb)))

        xb = self.max_pool2d(self.relu(self.conv3(xb)))

        xb = xb.view(bs,-1)

        xb = self.relu(self.clf1(xb))

        xb = self.dropout(xb)

        xb = self.clf2(xb)





        return xb



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

        train_loss = 0.0

        for xb, yb in train_dl:

            if(train_on_gpu):

                xb,yb = xb.cuda(),yb.cuda()

            item_loss , num_item = loss_batch(model, loss_func, xb, yb, opt)

            train_loss +=  num_item * item_loss



        model.eval()

        with torch.no_grad():

            if(train_on_gpu):

                losses_vad, nums_vad = zip(

                    *[loss_batch(model, loss_func, xb.cuda(), yb.cuda()) for xb, yb in valid_dl])

            else:

                losses_vad, nums_vad = zip(

                    *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])

        val_loss = np.sum(np.multiply(losses_vad , nums_vad )) / np.sum(nums_vad )

        train_loss = train_loss/len(train_dl.sampler)



        print('epoch {}, train_loss {}, val_loss {}'.format(epoch, train_loss, val_loss))
#network setting



model = Classifier()

if(train_on_gpu):

    model.cuda()



lr = 0.0005



#for orignal dataset, I use pos_weight.

#pos_weight = torch.tensor([5])

opt = torch.optim.Adam(model.parameters(), lr=lr)

loss_func = torch.nn.CrossEntropyLoss()



n_epoch = 10
train(n_epoch,model,loss_func,opt,train_dl,valid_dl)

model.eval()
if(train_on_gpu):

    ypred , yactual= zip(*[(model(xb.cuda()).detach().cpu().numpy(), yb.numpy()) for xb, yb in valid_dl])

else:

    ypred , yactual= zip(*[(model(xb).detach().numpy(), yb.numpy()) for xb, yb in valid_dl])



ypred = np.concatenate(ypred,axis=0)

yactual = np.concatenate(yactual,axis=0)



preds = np.argmax(ypred, axis=1)  

#print(preds)

#print(yactual)

print('Accuracy score: {}'.format(metrics.accuracy_score(yactual, preds)))

print('Confusion matrix: {}'. format(metrics.confusion_matrix(yactual, preds)))

#print('AUPRC score: {}'. format(metrics.average_precision_score(yactual, preds)))

#print('AUROC score: {}'.format(metrics.roc_auc_score(yactual, preds)))
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



xtest = test.values
yhat_test = []



for x in xtest:

    if(train_on_gpu):

        yhat = model(torch.tensor(x).float().unsqueeze(0).cuda()).detach().cpu().numpy()

    else:

        yhat = model(torch.tensor(x).float().unsqueeze(0)).detach().numpy()

    

    yhat_test.append(yhat)



yhat_test = np.concatenate(yhat_test,axis=0)





preds_test = np.argmax(yhat_test, axis=1) 