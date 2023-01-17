import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.model_selection import GridSearchCV
print(os.listdir("../input"))
data1=pd.read_csv('../input/column_2C_weka.csv')

data2=pd.read_csv('../input/column_3C_weka.csv')
data1.info()

data1.isnull().sum()

data1.head()
data_1=(data1['class']).unique()

sns.barplot(x=data_1,y=data1['class'].value_counts())

plt.xlabel('class')

plt.ylabel('values')

plt.grid()

plt.show()
data2.info()

data2.isnull().sum()

data2.head()
data_2=(data2['class']).unique()

sns.barplot(x=data_2,y=data2['class'].value_counts())

plt.xlabel('class')

plt.ylabel('values')

plt.grid()

plt.show()
data1.head()
data1['class']=pd.DataFrame(1 if each=='Normal' else 0 for each in data1['class'])

x_1=data1['class']

y_1=data1.drop(['class'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(y_1,x_1,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

print('{} data {}'.format(3,knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()

x,y = data1.loc[:,data1.columns != 'class'], data1.loc[:,'class']
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

steps = [('scalar', StandardScaler()),

         ('SVM', SVC())]

pipeline = Pipeline(steps)

parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)

cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)

cv.fit(x_train,y_train)



y_pred = cv.predict(x_test)



print("Accuracy: {}".format(cv.score(x_test, y_test)))

print("Tuned Model Parameters: {}".format(cv.best_params_))