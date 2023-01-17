# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import torchvision

from torch.utils.data import TensorDataset

from torch.optim import Adam, SGD



from sklearn.decomposition import PCA

import pylab



# Basic Numeric Computation

import numpy as np

import pandas as pd



# Look at data

import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt

import pandas as pd

#from math import pi

#from collections import Counter

import seaborn as sns

from sklearn.decomposition import PCA

import pylab

import time

from sklearn.manifold import TSNE

from sklearn import manifold

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_swiss_roll
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cpu")

epochs=10



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
y_viz_train = train['label']

X_viz_train = train.drop('label', axis=1)

X_viz_test = test



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_viz_train = scaler.fit_transform(X_viz_train)

X_viz_test = scaler.fit_transform(X_viz_test)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X_viz_train)

principalDf =pd.DataFrame(data = principalComponents, columns = ['principalcomponent1',  'principalcomponent2'])



label = pd.DataFrame(list(train['label']))

principalDf = pd.concat([principalDf,label],axis = 1,join='inner', ignore_index=True)

principalDf = principalDf.loc[:,~principalDf.columns.duplicated()]

principalDf.columns = ["principalcomponent1", "principalcomponent2", "label"] 
principalDf.head()
flatui = ["#9b59b6", "#3498db", "orange"]

sns.set_palette(flatui)

sns.lmplot( x="principalcomponent1", y="principalcomponent2", data=principalDf, fit_reg=False,

           hue='label', legend=False)



plt.figure(figsize=(13,10))
N = 10000

df_subset = X_viz_train[:N,:].copy()
time_start = time.time()

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(df_subset)

print('t-SNE done in {} seconds'.format(time.time()-time_start))
tsne3_subset = pd.DataFrame(columns=['tsne-3d-one', 'tsne-3d-two', 'tsne-3d-three'])



tsne3_subset['tsne-3d-one'] = tsne_results[:,0]

tsne3_subset['tsne-3d-two'] = tsne_results[:,1]

tsne3_subset['tsne-3d-three'] = tsne_results[:,2]
n_samples = 10000

X, color = make_swiss_roll(n_samples)
time_start = time.time()

tsne2d = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne2d_results = tsne.fit_transform(df_subset)

print('t-SNE done in {} seconds'.format(time.time()-time_start))
tsne2_subset = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])



tsne2_subset['tsne-2d-one'] = tsne2d_results[:,0]

tsne2_subset['tsne-2d-two'] = tsne2d_results[:,1]
fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(2, 1, 1,projection='3d')

ax.set_title('TSNE 3-d', fontsize=10)

ax.scatter(tsne_results[:,0],tsne_results[:,1],tsne_results[:,2],c = color,cmap="Accent",s=60)# we are picking up the x,y,z co-ordinate values from dataset

ax = fig.add_subplot(2, 1, 2) 

ax.set_title('TSNE - 2d', fontsize=10)

ax.scatter(tsne2d_results[:,0],tsne2d_results[:,1],c = color,cmap="Accent",s=60)
##############

######

#ISOMAP

######

##############



iso = manifold.Isomap(n_neighbors=6, n_components=2)

iso.fit(X)

manifold_iso_data = iso.transform(X)
fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(2, 1, 1,projection='3d')

ax.set_title('Here is the swiss roll maniflod', fontsize=10)

ax.scatter(X[:,0],X[:,1],X[:,2],c = color,cmap="Accent",s=60)

x = X[:,0][2:10000] 

y = X[:,1][2:10000] # Just as abovve, this time for column 1

ax.scatter(x,y,c = "black") #Now we randomly plot this in both 3D manifold (this may not be clearly visible as the  existing

ax.plot(x[2:4],y[2:4],c = "red")

ax = fig.add_subplot(2, 1, 2) 

#Now we plot 2D after ISOMAP...

ax.set_title('When compressing with ISOMAP', fontsize=10)

ax.scatter(manifold_iso_data[:,0],manifold_iso_data[:,1],c = color,cmap="Accent",s=60)

x = X[:,0][2:10000]#Now we plot the same 'black' samples, after ISOMAP in 2D and observe the distance in 2D.

y = X[:,1][2:10000]

ax.scatter(x,y,c = "black")

ax.plot(x[2:4],y[2:4],c = "red")

plt.show()
train['label'].head()
train.info()
train.describe()
test.info()
test.describe()
def Image_Data(raw: pd.DataFrame):

    y = raw['label'].values

    y.resize(y.shape[0],1)

    x = raw[[i for i in raw.columns if i != 'label']].values

    x = x.reshape([-1,1, 28, 28])

    y = y.astype(int).reshape(-1)

    x = x.astype(float)

    return x, y



## Convert to One Hot Embedding

def one_hot_embedding(labels, num_classes=10):

    y = torch.eye(num_classes) 

    return y[labels] 



x_train, y_train = Image_Data(train)

# Normalization

mean = x_train.mean()

std = x_train.std()

x_train = (x_train-mean)/std

# Numpy to Torch Tensor

x_train = torch.from_numpy(np.float32(x_train)).to(device)

y_train = torch.from_numpy(y_train.astype(np.long)).to(device)

y_train = one_hot_embedding(y_train)

#x_val = torch.from_numpy(np.float32(x_val))

#y_val = torch.from_numpy(y_val.astype(np.long))

# Convert into Torch Dataset

train_ds = TensorDataset(x_train, y_train)

train_dl = DataLoader(train_ds, batch_size=64)
def init_weights(m):

    if type(m) == nn.Linear:

        torch.nn.init.xavier_uniform(m.weight)

        m.bias.data.fill_(0.01)



## Flatten Later

class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)
# Train the network and print accuracy and loss overtime

def fit(train_dl, model, loss, optim, epochs=10):

    model = model.to(device)

    print('Epoch\tAccuracy\tLoss')

    accuracy_overtime = []

    loss_overtime = []

    for epoch in range(epochs):

        avg_loss = 0

        correct = 0

        total=0

        for x, y in train_dl: # Iterate over Data Loder

    

            # Forward pass

            yhat = model(x) 

            l = loss(y, yhat)

            

            #Metrics

            avg_loss+=l.item()

            

            # Backward pass

            optim.zero_grad()

            l.backward()

            optim.step()

            

            # Metrics

            _, original =  torch.max(y, 1)

            _, predicted = torch.max(yhat.data, 1)

            total += y.size(0)

            correct = correct + (original == predicted).sum().item()

            

        accuracy_overtime.append(correct/total)

        loss_overtime.append(avg_loss/len(train_dl))

        print(epoch,accuracy_overtime[-1], loss_overtime[-1], sep='\t')

    return accuracy_overtime, loss_overtime
def plot_accuracy_loss(accuracy, loss):

    f = pyplot.figure(figsize=(15,5))

    ax1 = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    ax1.title.set_text("Accuracy over epochs")

    ax2.title.set_text("Loss over epochs")

    ax1.plot(accuracy)

    ax2.plot(loss, 'r:')
ff_model = nn.Sequential(

    Flatten(),

    nn.Linear(28*28, 100),

    nn.ReLU(),

    nn.Linear(100, 10),

    nn.Softmax(1),

).to(device)

ff_model.apply(init_weights)



optim = Adam(ff_model.parameters())

loss = nn.MSELoss()

output = fit(train_dl, ff_model, loss, optim, epochs)

plot_accuracy_loss(*output)
index = 6

pyplot.imshow(x_train.cpu()[index].reshape((28, 28)), cmap="gray")

print(y_train[index])

class ConvNet(nn.Module):

    def __init__(self, num_classes=10):

        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7*7*32, num_classes)

        

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)

        return out


num_classes = 10

model = ConvNet(num_classes).to(device)

model.apply(init_weights)

loss = nn.MSELoss()

optim = SGD(model.parameters(), lr=0.003, momentum=0.9)

plot_accuracy_loss(*fit(train_dl, model,loss,optim,epochs))
x_test = test.values

x_test = x_test.reshape([-1, 28, 28]).astype(float)

x_test = (x_test-mean)/std

x_test = torch.from_numpy(np.float32(x_test))

x_test.shape

def export_csv(model_name, predictions):

    df = pd.DataFrame(prediction.tolist(), columns=['Label'])

    df['ImageId'] = df.index + 1

    file_name = f'submission_{model_name}.csv'

    print('Saving ',file_name)

    df[['ImageId','Label']].to_csv(file_name, index = False)



ff_test = ff_model(x_test.float())

prediction = torch.argmax(ff_test,1)

print('Prediction',prediction)

export_csv('ff_model',prediction)

torch.save(model.state_dict(), 'model_ff.ckpt')