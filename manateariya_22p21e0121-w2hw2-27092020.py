# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import recall_score,precision_score,f1_score

from sklearn.neural_network import MLPClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
v=pd.read_csv("/kaggle/input/titanic/train.csv")

v.columns

v.head()

from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

kf
data=pd.read_csv("/kaggle/input/titanic/train.csv")

data.shape
import math

data_train=[]

data_test=[]

arr_data=[]

arr_sur=[]

for i in range (data.shape[0]):

    d=[]

    d.append(data["PassengerId"][i])

    d.append(data["Pclass"][i])

    if (data["Sex"][i]=='male'):

        d.append(1)

    if(data["Sex"][i]=='female'):

        d.append(0)

    if(math.isnan(data["Age"][i])):

        d.append(0)

    else:

        d.append(data["Age"][i])

    d.append(data["SibSp"][i])

    d.append(data["Parch"][i])

    d.append(data["Fare"][i])

    arr_data.append(d)

    arr_sur.append(data["Survived"][i])

    

    
arr_data
data1train=[]

data1test=[]

y1=[]

ytest1=[]

data2train=[]

data2test=[]

y2=[]

ytest2=[]

data3train=[]

data3test=[]

y3=[]

ytest3=[]

data4train=[]

data4test=[]

y4=[]

ytest4=[]

data5train=[]

data5test=[]

y5=[]

ytest5=[]



datafortrain=[data1train,data2train,data3train,data4train,data5train]

datafortest=[data1test,data2test,data3test,data4test,data5test]

ylis=[y1,y2,y3,y4,y5]

ytes=[ytest1,ytest2,ytest3,ytest4,ytest5]

kf.split(arr_data)

count=0

for train_index, test_index in kf.split(arr_data):

    #print(len(train_index), len(test_index))

    for i in train_index:

        datafortrain[count].append(arr_data[i])

        ylis[count].append(arr_sur[i])

    for r in test_index:

        datafortest[count].append(arr_data[r])

        ytes[count].append(arr_sur[r])

    count=count+1

        

len(data3train)
def decisiontr(x,y,xtest,ytest):

    print("Decision tree")

    tree1=tree.DecisionTreeClassifier()

    model1 = tree1.fit(x, y)

    predict=model1.predict(xtest)

    recall = recall_score(ytest,predict)

    precision = precision_score(ytest,predict)

    f_measure = f1_score(ytest,predict)

    print("Recall:",recall,"\nPrecision:",precision,"\nF measure",f_measure)

    return(f_measure)

def naivebay(x,y,xtest,ytest):

    print("Naive Bay")

    gnb = GaussianNB()

    model2= gnb.fit(x, y)

    predict=model2.predict(xtest)

    recall = recall_score(ytest,predict)

    precision = precision_score(ytest,predict)

    f_measure = f1_score(ytest,predict)

    print("Recall:",recall,"\nPrecision:",precision,"\nF measure",f_measure)

    return(f_measure)

def neuronnet(x,y,xtest,ytest):

    print("Neuron Network")

    clf = MLPClassifier()

    model3= clf.fit(x, y)

    predict=model3.predict(xtest)

    recall = recall_score(ytest,predict)

    precision = precision_score(ytest,predict)

    f_measure = f1_score(ytest,predict)

    print("Recall:",recall,"\nPrecision:",precision,"\nF measure",f_measure)

    return(f_measure)
f1=0

f2=0

f3=0

for i in range (5):

    print("----------------------------------------Data Set:",i+1,"-----------------------------------------------")

    f1=decisiontr(datafortrain[i],ylis[i],datafortest[i],ytes[i])

    f2=naivebay(datafortrain[i],ylis[i],datafortest[i],ytes[i])

    f3=neuronnet(datafortrain[i],ylis[i],datafortest[i],ytes[i])

    print("Average F-measure:",(f1+f2+f3)/3)