

import numpy as np 
import pandas as pd 
import random as rnd

import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

df_trainingSet=pd.read_csv('../input/iris.arff.csv');


df_trainingSet.head()


df_trainingSet.tail()


df_trainingSet.info()


df_trainingSet.describe()


df_trainingSet.describe(include=['O'])





g = sns.FacetGrid(df_trainingSet, col='class')


g.map(plt.hist, 'sepallength', bins=30)



g = sns.FacetGrid(df_trainingSet, col='class')
g.map(plt.hist,'sepalwidth', bins=40)


g = sns.FacetGrid(df_trainingSet, col='class')
g.map(plt.hist,'petallength',bins=40)


g = sns.FacetGrid(df_trainingSet, col='class')
g.map(plt.hist,'petalwidth',bins=40)






df_input_Data=df_trainingSet.drop("class",axis=1)

df_tr_class = df_trainingSet["class"]


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df_input_Data,df_tr_class,test_size=0.3,random_state=1)

model=DecisionTreeClassifier()

train=model.fit(x_train,y_train)
train

prediction=model.predict(x_test)
prediction

precision=train.score(x_test,y_test)
precision*100


x_train,x_test,y_train,y_test=train_test_split(df_input_Data,df_tr_class,test_size=0.3,random_state=1)

model=GaussianNB()

train=model.fit(x_train,y_train)
train

prediction=model.predict(x_test)
prediction

precision=train.score(x_test,y_test)
precision*100



x_train,x_test,y_train,y_test=train_test_split(df_input_Data,df_tr_class,test_size=0.3,random_state=1)

model=Perceptron()

train=model.fit(x_train,y_train)
train

prediction=model.predict(x_test)
prediction

precision=train.score(x_test,y_test)
precision*100
