# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import time

t0 = time.time()



import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



dataset = pd.read_csv("../input/adult-census-dataset/adult.csv")

dataset.describe()
dataset.head(5)
string_columns=dataset.columns

print(len(dataset.columns))

string_columns=[]

int_columns=[]

for i in dataset.columns:

    if(type(dataset[i][0]) !=np.int64 and i!='salary'):

        string_columns.append(i)

    else:

        int_columns.append(i)

dataset.hist(histtype='stepfilled',figsize = (20,20))
fig, ax =plt.subplots(4,2)

for i in range(len(string_columns)-1):

    sns.countplot(dataset[string_columns[i]], ax=ax[i//2][i%2])

fig.set_size_inches(18.5, 10.5)

fig.show()
sns.countplot(x=string_columns[0], hue="salary", data=dataset)

plt.show()
dataset['salary']=(dataset['salary']==' <=50K').astype(np.int64)
y=dataset['salary']
dataset


one_hot_coloumns = []

for i in string_columns:

    for j in dataset[i].unique():

        

        if(j==' ?'):

            continue

        one_hot_coloumns.append(j)

        print(j)

        dataset[j] = (dataset[i] == j).astype(int)
print(len(one_hot_coloumns))
print(one_hot_coloumns)
len(dataset.columns)
dataset.head(5)
dataset=dataset.drop(string_columns,axis=1)
def csv_normlize(data,attr1,attr):

    data[attr1] = (data[attr] - data[attr].min() ) / ( data[attr].max() - data[attr].min() )

for i in int_columns:

    csv_normlize(dataset,i,i)
dataset.columns
y = np.array(dataset['salary']).astype(float)

x = np.array(dataset.drop('salary',axis=1)).astype(float)



def split(x,y,test_size):

    tot=len(x)

    t1=int(test_size*tot)

    arr = np.arange(tot)

    np.random.shuffle(arr)

    idtrain=arr[t1:]

    idtest=arr[:t1]

    print(len(idtrain),len(idtest),tot)

    print(idtrain,idtest)

    return x[idtrain],x[idtest],y[idtrain],y[idtest]



#Splitting the data into Train and Test

xtrain, xtest, ytrain, ytest = split(x,y,1/3)



ytrain
print(y.sum()/len(y))
def update(b,a,X,Y,learning_rate):

    db =[0]*len(b)

    da = 0

    tot=len(X)

    for i in range(tot):

        q=0

        lda=3-2*y[i]

        for j in range(len(b)):

            q+=b[j]*X[i][j]

        p=2*(q+a-Y[i])*lda*lda

        for j in range(len(b)):

            db[j]+= p*X[i][j]

        da += p

    for j in range(len(b)):

        b[j] -= db[j]/float(tot) * learning_rate

    a -= da/float(tot) * learning_rate

    return b,a



def calc(x,y):

    return sum(((y-x)*(3-y*2))**2)/len(x)

  



b=[0]*len(xtrain[0])

a=1

def sigmoid(x): 

    return 1/(1+np.exp(-x))

def prdict(x,y):

    sig=sigmoid(x*2-1)

    return ((y==(sig>0.5)).astype(int).sum())

    
cost=[]

acc=[]



learning_rate = 0.1





for i in range(100):

    lr=np.sqrt(i+1)

    lr=learning_rate/lr

    



    b,a = update(b,a,xtrain,ytrain,lr)

    

    pred=np.zeros(len(x))

    for j in range(len(x)):

        q=a

        for k in range(len(b)):

            q+=x[j][k]*b[k]

        pred[j]=q

    print(pred)

    c=calc(pred,y)

    cost.append(c)

    p=prdict(pred,y)/len(y)

    acc.append(p)

    print("Loss =",c)

    print("acc =",p)
for i in range(100):

    lr=np.sqrt(i+1)

    lr=learning_rate/lr

    



    b,a = update(b,a,xtrain,ytrain,lr)

    

    pred=np.zeros(len(x))

    for j in range(len(x)):

        q=a

        for k in range(len(b)):

            q+=x[j][k]*b[k]

        pred[j]=q

    print(pred)

    c=calc(pred,y)

    cost.append(c)

    p=prdict(pred,y)/len(y)

    acc.append(p)

    print("Loss =",c)

    print("acc =",p)
plt.plot(range(len(cost)),cost,color = 'r')

plt.plot(range(len(acc)),cost,color = 'r')



pred=np.zeros(len(x))

for j in range(len(x)):

    q=a

    for k in range(len(b)):

        q+=x[j][k]*b[k]

    pred[j]=q

    pred[j]=(pred[j]>0.5)*1



hmap=np.zeros((2,2))

for i in range(len(x)): hmap[int(y[i])][int(pred[i])]+=1



print(hmap)
sns.heatmap(pd.DataFrame(hmap), annot=True, cmap="YlGnBu" ,fmt='g')

plt.xlabel('Predicted label')

plt.ylabel('True label')
t1 = time.time()

print(t1-t0, "seconds wall time")