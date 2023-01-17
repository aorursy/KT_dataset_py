import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_iris,load_breast_cancer

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.metrics import accuracy_score,f1_score

from sklearn.base import BaseEstimator,ClassifierMixin

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

np.set_printoptions(suppress=True)
Xiris,Yiris = load_iris(return_X_y=True)

Xiris = Xiris[:,2:]

Xiris = MinMaxScaler().fit_transform(Xiris)

plt.scatter(Xiris[:,0],Xiris[:,1],c=Yiris);
class Net(nn.Module,BaseEstimator,ClassifierMixin):

    def __init__(self,labels):

        super().__init__()

        self.labels = labels

        self.fc1 = nn.Linear(2,16)

        self.fc2 = nn.Linear(16,labels)

    def forward(self,x):

        x = F.relu(self.fc1(x))

        return self.fc2(x)

    

    def fit(self,X,Y):

        for module in self.children():

            module.reset_parameters()

            

        opt = optim.Adam(model.parameters())

        X,Y = torch.from_numpy(X.copy()),torch.from_numpy(Y.copy())

        for i in range(4001):

            opt.zero_grad()

            out = model(X)

            loss = F.cross_entropy(out,Y)

            loss.backward()

            opt.step()

        return self

            

    def predict(self,X):

        with torch.no_grad():

            out = model(torch.from_numpy(X.copy())).numpy()

        return np.argmax(out,axis=1)

        

model = Net(labels=3).double()
res = cross_validate(model,Xiris,Yiris,cv=3)

res['test_score'].mean(),res['test_score'].std()
xp = np.arange(-.2,1.2,.01)

yp = np.arange(-.2,1.2,.01)

xx,yy = np.meshgrid(xp,yp) 



def plot_decision_boundary(model,X,Y,interval=.02):

    fig, ax = plt.subplots()

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=.99);

    ax.scatter(X[:,0],X[:,1],c=Y);

    

plt.rcParams["animation.html"] = "jshtml"

import matplotlib.animation



def plot_single_instance(model,history):

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,5))

    plt.close()



    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax1.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=.99);

    sc = bar = None

    def animate(i):

        nonlocal sc,bar

        xa,log = history[i]

        if sc: sc.remove()

        sc = ax1.scatter(xa[:,0],xa[:,1],c='yellow',edgecolors='black');

    

        log = log.squeeze()

        labels = len(log)

        y_pos = np.arange(labels)

        if bar: bar.remove()

        bar = ax2.bar(y_pos,log,color='blue');

        ax2.set_xticks(y_pos)

        ax2.set_xticklabels([str(i) for i in range(labels)]);



    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(history),interval=350)

    return ani



def plot_dataset_movement(model,history,step=2):

    fig,ax = plt.subplots(figsize=(8,5))

    plt.close()



    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=.99);

    sc = None

    def animate(i):

        nonlocal sc,step

        if i%step!=0: return

        xa,log = history[i]

        if sc: sc.remove()

        sc = ax.scatter(xa[:,0],xa[:,1],c='yellow',edgecolors='black');



    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(history),interval=350)

    return ani
def cw_attack(model,x,y,lr=.001): #both tensors

    x.requires_grad_(True)

    with torch.no_grad():

        logits = model(x)

        IMX = torch.argmax(logits,dim=1) #Original classes



    history = {}

    i = 0

    while True:

        logits = model(x)

        history[i] = (x.detach().clone().numpy(),F.softmax(logits,dim=1).detach().clone().numpy())

        maxs,imx = logits.max(dim=1)

        unbroken = (imx==IMX).double()

        #print(i,unbroken.sum().item(),unbroken.shape)

        if unbroken.sum()==0: 

            break



        y = logits.clone()

        y[range(len(y)),imx] = -99

        sec_maxs,_ = y.max(dim=1) 

        loss = (maxs-sec_maxs).mean()

        loss.backward()

        

        #print(i,maxs,sec_maxs)

        tmp = x.grad.data*unbroken.view(-1,1)

        #print(tmp,unbroken)

        x.data.sub_(lr*tmp)

        x.grad.data.zero_()

        i+=1

    return history

model.fit(Xiris,Yiris)

plot_decision_boundary(model,Xiris,Yiris,.01)
idx,N = 75,10

x,y = torch.from_numpy(Xiris[idx:idx+N].copy()),torch.from_numpy(Yiris[idx:idx+N].copy())

history = cw_attack(model,x,y,.002)

len(history)
plot_dataset_movement(model,history,2)
X,Y = load_breast_cancer(return_X_y=True)

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2).fit(X)

Xp = pca.transform(X)

Xp = MinMaxScaler().fit_transform(Xp)

pca.explained_variance_ratio_,pca.explained_variance_ratio_.sum()
plt.scatter(Xp[:,0],Xp[:,1],c=Y);
res = cross_validate(model,Xp.copy(),Y.copy(),cv=3)

res['test_score'].mean(),res['test_score'].std()
model = Net(labels=2).double()

model.fit(Xp,Y)

plot_decision_boundary(model,Xp,Y,.01)
idx,N = 175,1

x,y = torch.from_numpy(Xp[idx:idx+N].copy()),torch.from_numpy(Y[idx:idx+N].copy())

history = cw_attack(model,x,y)

print(len(history))

plot_single_instance(model,history)
idx,N = 175,10

x,y = torch.from_numpy(Xp[idx:idx+N].copy()),torch.from_numpy(Y[idx:idx+N].copy())

history = cw_attack(model,x,y)

print(len(history))

plot_dataset_movement(model,history,2)