import numpy as np

import sklearn.datasets

import pandas as pd

import matplotlib.pyplot as plt

import random
breast_cancer= sklearn.datasets.load_breast_cancer()
X=breast_cancer.data

Y=breast_cancer.target

type(X)
X.shape,Y.shape
data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)

data.head()
data['class']=breast_cancer.target
data.shape

data.head()
data.dtypes

data.describe()
data['class'].value_counts()
#data.groupby('class').mean()

data.groupby('class').max()
from sklearn.model_selection import train_test_split
X=data.drop('class',axis=1)

Y = data['class']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state = 1)
X_train.shape,Y_train.shape,X_test.shape
type(X_train),type(Y_train)
X.shape,Y.shape
Y.mean(),Y_train.mean(),Y_test.mean()
X_train.mean(),X_test.mean(),X.mean()
type(X_train)
plt.plot(X_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()
X_binarized_train=X_train.apply(pd.cut,bins =2,labels=[1,0])
plt.plot(X_binarized_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()
X_binarized_test=X_test.apply(pd.cut,bins =2,labels=[1,0])
plt.plot(X_binarized_test.T,'*')

plt.xticks(rotation='vertical')

plt.show()
X_binarized_train = np.asarray(X_binarized_train)

X_binarized_test=np.asarray(X_binarized_test)
type(X_binarized_test),type(X_binarized_train)
b=3

i=100

if (np.sum(X_binarized_train[100,]))>=b:

  print("MP neuron inference is melignan") #melignan = has breast cancer

else:

  print("MP neuron inference is benign")

if (Y_train[i]==1):

  print('ground truth is melignan')

else:

  print("ground truth is benign")    
Y_pred=[]

accurate_rows=0

for x,y in zip(X_binarized_train,Y_train):

  y_pred=(np.sum(x)>=b)

  Y_pred.append(y_pred)

  accurate_rows +=(y==y_pred) 

print(accurate_rows)  

print(accurate_rows/X_binarized_train.shape[0])
for b in range(X_binarized_train.shape[1]+1):

  Y_pred_train=[]

  accurate_rows=0

  for x,y in zip(X_binarized_train,Y_train):

    y_pred=(np.sum(x)>=b)

    Y_pred_train.append(y_pred)

    accurate_rows +=(y==y_pred)

  print(b, accurate_rows/X_binarized_train.shape[0])  
from sklearn.metrics import accuracy_score
b=27

Y_pred_test=[]

accurate_rows=0

for x in X_binarized_test:

  y_pred=(np.sum(x)>=b)

  Y_pred_test.append(y_pred)

accuracy=accuracy_score(Y_test,Y_pred_test)

print(accuracy)
class mpneuron:

  def __init__(self):

    self.b=None



  def model(self,x):

    return(sum(x)>=self.b) 



  def predict(self,X):

    Y=[]

    for x in X:

      result=self.model(x)

      Y.append(result)

    return np.array(Y)



  def fit(self,X,Y):

    accuracy= {}



    for b in range(X.shape[1]+1):

      self.b=b

      Y_pred=self.predict(X)

      accuracy[b]=accuracy_score(Y_pred,Y)

    best_b=max(accuracy,key=accuracy.get)

    self.b=best_b

    print("optmal value of b is",best_b)

    print("Optimal accuracy is",accuracy[best_b])
Mpneuron = mpneuron()
Mpneuron.fit(X_binarized_train,Y_train)
Y_pred_test=Mpneuron.predict(X_binarized_test)

accurac=accuracy_score(Y_pred_test,Y_test)
print(accurac)
type(X_train)
class perceptron:

  def __init__(self):

    self.w = None

    self.b = None



  def model(self,x):

    return 1 if (np.dot(self.w,x)>=self.b) else 0



  def predict(self,X):

    Y = []

    for x in X:

      result = self.model(x)

      Y.append(result)

    return np.array(Y)

      

  def fit(self,X,Y):

    

    self.w = np.ones(X.shape[1])

    self.b=0

    for x,y in zip(X,Y):

      y_pred=self.model(x)

      if y==1 and y_pred == 0:

        self.w = self.w+x

        self.b = self.b+1

      elif y==0 and y_pred==1:

        self.w=self.w-x

        self.b = self.b-1
Perceptron= perceptron()
X_train=np.asarray(X_train)

type(X_train)
X_test=np.asarray(X_test)

type(X_test)
Perceptron.fit(X_train,Y_train)
plt.plot(Perceptron.w)

plt.show()
Y_pred_train=Perceptron.predict(X_train)

Y_pred_train.shape
accuracy=accuracy_score(Y_pred_train,Y_train)
print(accuracy)
Y_pred_test=Perceptron.predict(X_test)

Y_test.shape
accuracy=accuracy_score(Y_pred_test,Y_test)
print(accuracy)

class perceptron:

  def __init__(self):

    self.w = None

    self.b = None



  def model(self,x):

    return 1 if (np.dot(self.w,x)>=self.b) else 0



  def predict(self,X):

    Y = []

    for x in X:

      result = self.model(x)

      Y.append(result)

    return np.array(Y)

      

  def fit(self,X,Y,epoch=1):

    

    accuracy = []

    max_accuracy = 0

    self.w = np.ones(X.shape[1])

    self.b=0

    for i in range(epoch):  

      for x,y in zip(X,Y):

        y_pred=self.model(x)

        if y==1 and y_pred == 0:

          self.w = self.w+x

          self.b = self.b+1

        elif y==0 and y_pred==1:

          self.w=self.w-x

          self.b = self.b-1

      accuracy.append(accuracy_score(self.predict(X),Y))

      if(accuracy[i]>max_accuracy):

        max_accuracy=accuracy[i]

        chkptw = self.w

        chkptb = self.b



    #self.w=chkptw #checkpointing for storing vauels of parametrs here w and b for which accuray is higher for later use

    #self.b=chkptb



    print(max_accuracy)      

    plt.plot(accuracy)

    plt.show()       
Perceptron= perceptron()
Perceptron.fit(X_train,Y_train,100)
Y_pred_train=Perceptron.predict(X_train)

accuracy=accuracy_score(Y_pred_train,Y_train)

print(accuracy)
class perceptron:

  def __init__(self):

    self.w = None

    self.b = None



  def model(self,x):

    return 1 if (np.dot(self.w,x)>=self.b) else 0



  def predict(self,X):

    Y = []

    for x in X:

      result = self.model(x)

      Y.append(result)

    return np.array(Y)

      

  def fit(self,X,Y,epoch=1,lr=1):

    

    accuracy = []

    w_matrix=[]

    max_accuracy = 0

    self.w = np.ones(X.shape[1])

    self.b=0

    for i in range(epoch):  

      for x,y in zip(X,Y):

        y_pred=self.model(x)

        if y==1 and y_pred == 0:

          self.w = self.w+x+lr*x

          self.b = self.b+1+lr*1

        elif y==0 and y_pred==1:

          self.w=self.w-x-lr*x

          self.b = self.b-1-lr*1

      w_matrix.append(self.w)    

      accuracy.append(accuracy_score(self.predict(X),Y))

      if(accuracy[i]>max_accuracy):

        max_accuracy=accuracy[i]

        chkptw = self.w

        chkptb = self.b



    self.w=chkptw #checkpointing for storing vauels of parametrs here w and b for which accuray is higher for later use

    self.b=chkptb



    print(max_accuracy)      

    plt.plot(accuracy)

    plt.show()



    return np.array(w_matrix)       
Perceptron= perceptron()
Perceptron.fit(X_train,Y_train,100,0.01)
Y_pred_train=Perceptron.predict(X_train)

accuracy=accuracy_score(Y_pred_train,Y_train)

print(accuracy)
Y_pred_test=Perceptron.predict(X_test)

accuracy=accuracy_score(Y_pred_test,Y_test)

print(accuracy)