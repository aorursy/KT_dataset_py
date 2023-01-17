#Chargement des données :

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold

df_trainingSet = pd.read_csv('../input/iris.arff.csv')






df_trainingSet.head()
df_trainingSet.tail()
df_trainingSet.info()
df_trainingSet.describe()
df_trainingSet.describe(include=['O'])
g = sns.FacetGrid(df_trainingSet, col='class')
g.map(plt.hist,'sepallength')
g = sns.FacetGrid(df_trainingSet, col='class')
g.map(plt.hist,'sepalwidth')
g = sns.FacetGrid(df_trainingSet, col='class')
g.map(plt.hist,'petallength')
g = sns.FacetGrid(df_trainingSet, col='class')
g.map(plt.hist,'petalwidth')
df_input_Data=df_trainingSet.drop("class",axis=1)
df_tr_class = df_trainingSet["class"]
arbre = DecisionTreeClassifier()
arbre.fit(df_input_Data, df_tr_class)
pre_arbre = round(arbre.score(df_input_Data, df_tr_class) * 100, 2)
pre_arbre

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df_input_Data,df_tr_class,test_size=0.3,random_state=1)

model=DecisionTreeClassifier()

data2=model.fit(x_train,y_train)
data2

predict=model.predict(x_test)
predict

data3=data2.score(x_test,y_test)
data3*100
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df_input_Data,df_tr_class,test_size=0.3,random_state=1)

model=GaussianNB()

train=model.fit(x_train,y_train)
train

prediction=model.predict(x_test)
prediction

precision=train.score(x_test,y_test)
precision*100
bayes = GaussianNB()
bayes.fit(df_input_Data, df_tr_class)
pre_bayes = round(bayes.score(df_input_Data, df_tr_class) * 100, 2)
pre_bayes
perceptron = Perceptron()
perceptron.fit(df_tr_inputData, df_tr_class)
acc_perceptron = round(perceptron.score(df_tr_inputData, df_tr_class) * 100, 2)
acc_perceptron