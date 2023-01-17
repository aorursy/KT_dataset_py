# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cardiovascular-disease-dataset/"))

dataset = pd.read_csv("../input/cardiovascular-disease-dataset/cardio_train.csv",sep=";")

column=["id","cardio"]

x=dataset.drop(column,axis=1)

y=dataset["cardio"]

y=y.astype(int)

print(dataset.isnull().sum())
print(dataset.info())
print(dataset.head(10))

dataset.describe()
rcParams['figure.figsize'] = 10,5

sns.countplot(x='gender',hue='cardio',data=dataset)
dataset.cardio.value_counts()

diseased=(len(dataset[dataset.cardio==1])/len(dataset.cardio))*100

diseased_male=len(dataset[(dataset.cardio==1) & (dataset.gender==1)])/len(dataset.cardio)*100

diseased_female=len(dataset[(dataset.cardio==1) & (dataset.gender==2)])/len(dataset.cardio)*100



print("{:.2f}% of the total count were diseased, amoung which {:.2f}% were male and {:.2f}% were female".format(diseased,diseased_male,diseased_female))



non_diseased=(len(dataset[dataset.cardio==0])/len(dataset.cardio))*100

non_diseased_male=len(dataset[(dataset.cardio==0) & (dataset.gender==1)])/len(dataset.cardio)*100

non_diseased_female=len(dataset[(dataset.cardio==0) & (dataset.gender==2)])/len(dataset.cardio)*100

print("\n{:.2f}% of the total count were  not diseased, amoung which {:.2f}% were male and {:.2f}% were female".format(non_diseased,non_diseased_male,non_diseased_female))

pd.crosstab(dataset.gender, dataset.cardio).plot(kind="bar",figsize=(10,5),color=['#1CA53B','#AA1111' ])

plt.title("frequency of disease vs gender")

plt.ylabel('range')

plt.xlabel('male vs female')

plt.legend(["Diseased","Not Diseased"])

plt.show()
sns.heatmap(dataset.corr(),annot=True,cmap='RdYlGn') 

#sns.heatmap(dataset.corr(),annot=True)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show
sns.distplot(dataset["gender"])
col=['cholesterol','gluc', 'smoke', 'alco', 'active']

data_value=pd.melt(dataset,id_vars="cardio",value_vars=dataset[col])

sns.catplot(x="variable",hue="value",col="cardio",data=data_value,kind="count")
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.2,random_state=0)
Classifiers = list()

from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()

GNB.fit(train_x,train_y)

y_pred_NB = GNB.predict(test_x)

Classifiers.append(y_pred_NB)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors = 5)

knn.fit(train_x,train_y)

y_pred_KN = knn.predict(test_x)

Classifiers.append(y_pred_KN)
from sklearn.tree import DecisionTreeClassifier

DTR = DecisionTreeClassifier()

DTR.fit(train_x,train_y)

y_pred_Deci = DTR.predict(test_x)

Classifiers.append(y_pred_Deci)
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=500,random_state=0)

RFC.fit(train_x,train_y)

y_pred_RF = RFC.predict(test_x)

Classifiers.append(y_pred_RF)
Class = ['Naive Bayes' , 'KNeighbors' ,'DecisionTree', 'RandomForest' ]

score=list()

a=0

index=0

from sklearn.metrics import accuracy_score

for pred in range(len(Classifiers)):

    if a < accuracy_score(test_y,Classifiers[pred]):

        a = accuracy_score(test_y,Classifiers[pred])

        index=pred

        

    print("accuracy of {} classifier is {:.2f}%".format(Class[pred],accuracy_score(test_y,Classifiers[pred])*100))

    

print("\nbest classifier is {} and the accuracy is {:.2f}%".format(Class[index],a*100))