# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

data=data.sample(10000)



print(data.shape)

data.head()
import seaborn as sns

sns.countplot(data['Class'],label="Class")

plt.show()



data['Class'].value_counts()
print(data.columns[1:-1])

print(data.columns[-1:])

print(data.columns[:-1].values.tolist())


from matplotlib import cm

feature_names = data.columns[1:-1].values.tolist()

X = data[feature_names].values

y = data['Class'].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

for i in y_train:

    if i==1:

        print(i)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(logreg.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(logreg.score(X_test, y_test)))
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
def sigmoid(z):

     return (1/(1+np.exp(-z)))

sigmoid(0)

def compareV(v1,v2,l):

    eq=0

    for i in range(l):

        if(v1[i]==v2[i]):

            eq+=1

    return eq
np.random.seed(1)

alpha = 1e-3

min_diff = 2e-5

data_n=X_train.shape[0]

data_m=X_train.shape[1]

#W = np.random.randn(data_n,data_m) * 0.01

thetas = np.zeros((data_m,1))

b = np.zeros((data_n,1))

print(data_n,data_m)

print(X_train.shape,y_train.shape,y_train[1])

count=1

lastE=np.zeros((data_m,1))

while(1):

    e=np.zeros((data_m,1))  

    for i in range(data_n):

        z=0

        for j in range(data_m):

            z+=thetas[j]*X_train[i][j]

        h=sigmoid(z)

        for j in range(data_m):

            e[j]+=(y_train[i]-h)*X_train[i][j]

    th_new=np.zeros((data_m,1))

    for j in range(data_m):

            th_new[j]=thetas[j]+alpha*e[j]

    print('iteration no '+str(count),'error = ',np.sum(e),np.sum(lastE),'diff= ',abs(np.sum(e)-np.sum(lastE)))

    if(abs(np.sum(e)-np.sum(lastE))<=min_diff):

        #print(th_new,thetas,e)

        break;

    else:

        count+=1

        for j in range(data_m):

            thetas[j]=th_new[j]

        lastE=np.copy(e)

    

print('done')

#print(sums.shape,b.shape)

#print(X_train.shape,y_train.shape,W.shape,sigmoid(np.dot(W,X_train.T)).shape)
np.random.seed(1)

alpha = 1e-3

min_diff = 2e-5

data_n=X_train.shape[0]

data_m=X_train.shape[1]

#W = np.random.randn(data_n,data_m) * 0.01

thetas = np.zeros((data_m,1))

b = np.zeros((data_n,1))

print(data_n,data_m)

print(X_train.shape,y_train.shape,y_train[1])

count=1

lastE=np.zeros((data_m,1))

while(1):

    e=np.zeros((data_m,1))  

    for i in range(data_n):

        z=0

        for j in range(data_m):

            z+=thetas[j]*X_train[i][j]

        h=sigmoid(z)

        for j in range(data_m):

            e[j]+=(y_train[i]-h)*X_train[i][j]

    th_new=np.zeros((data_m,1))

    for j in range(data_m):

            th_new[j]=thetas[j]+alpha*e[j]

    print('iteration no '+str(count),'error = ',np.sum(e),np.sum(lastE),'diff= ',abs(np.sum(e)-np.sum(lastE)))

    if(abs(np.sum(e)-np.sum(lastE))<=min_diff):

        #print(th_new,thetas,e)

        break;

    else:

        count+=1

        for j in range(data_m):

            thetas[j]=th_new[j]

        lastE=np.copy(e)

    

print('done')

#print(sums.shape,b.shape)

#print(X_train.shape,y_train.shape,W.shape,sigmoid(np.dot(W,X_train.T)).shape)
test_n=X_test.shape[0]

test_m=X_test.shape[1]

print(test_n,test_m)

predict = np.zeros((test_n,1))

for i in range(test_n):

    z=0

    for j in range(data_m):

        z+=thetas[j]*X_test[i][j]

    h=sigmoid(z)

    predict[i]=h

predictF=np.where(predict<0.5,0,1)

unique, counts = np.unique(predictF, return_counts=True)

#dict(zip(unique, counts))

eq=compareV(predictF,y_test,test_n)

print(eq,X_test.shape,y_test.shape,predictF.shape)

print('test accuracy= ',eq/test_n)
train_n=X_train.shape[0]

train_m=X_train.shape[1]

print(train_n,train_m)

predict = np.zeros((train_n,1))

for i in range(train_n):

    z=0

    for j in range(data_m):

        z+=thetas[j]*X_train[i][j]

    h=sigmoid(z)

    predict[i]=h

predictF=np.where(predict<0.5,0,1)

unique, counts = np.unique(predictF, return_counts=True)

#dict(zip(unique, counts))

eq=compareV(predictF,y_train,train_n)

print(eq,X_train.shape,y_train.shape,predictF.shape)

print('train accuracy= ',eq/train_n)
np.random.seed(1)

alpha = 1e-3

min_diff = 2e-5

data_n=X_train.shape[0]

data_m=X_train.shape[1]

#W = np.random.randn(data_n,data_m) * 0.01

thetas = np.zeros((data_m,))

b = np.zeros((data_n,1))

print(data_n,data_m)

print(X_train.shape,y_train.shape,y_train[2])

count=1

while(1):

    e=np.zeros((data_m,1))

    for i in range(data_n):

        z=0         

        z+=np.dot(thetas,X_train[i])

        h=sigmoid(z)

        for j in range(data_m):

           # if j==0:

         #       print((y_train[i]).shape,h.shape,X_train[i][j].shape)

            e[j]+=(y_train[i]-h)*X_train[i][j]

    #th_new=np.zeros((data_m,1))   

    th_new[:]=(thetas[:])+np.reshape((alpha*e[:]),(30,))

    print('iteration no '+str(count),'error = ',np.sum(e))

    if(abs(np.sum(e))<=min_diff):

        #print(th_new,thetas,e)

        break;

    else:

        count+=1

      #  print(thetas.shape)

        thetas[:]=th_new[:]

      #  print(thetas.shape)



    

print('done')

#print(sums.shape,b.shape)

#print(X_train.shape,y_train.shape,W.shape,sigmoid(np.dot(W,X_train.T)).shape)