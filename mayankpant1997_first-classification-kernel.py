import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns
orig_data = pd.read_csv(r'../input/salary_data.csv', 

                        na_values=' ?')

data = pd.DataFrame(orig_data)

data.head()
data.shape
#checking for null values

data.isnull().sum()
data.dropna(inplace=True)

data.head()
data.shape
data.head()
#age

ax=sns.violinplot(data['salary'], data['age'])

ax.set(xlabel='salary',ylabel='age')



'''

<=50k are aged around 25 years

>50k are aged around 40 to 50 years



'''
#fnlwgt

ax=sns.violinplot(data['salary'], data['fnlwgt'])

ax.set(xlabel='salary',ylabel='fnlwgt')



'''

there is no difference in the salary

so fnlwgt in not affecting it

'''
ax=sns.violinplot(data['salary'], data['hours-per-week'])

ax.set(xlabel='salary',ylabel='hours per week')



'''

here we cannot really say how much a person earns

based on his hours per week

because both categories have quite similar distribution



so we dont include it

'''

#marital status

data.groupby(['marital-status'])['salary'].value_counts(normalize=True)
#education

data.groupby(['education'])['salary'].value_counts(normalize=True)
#workclass

data.groupby(['workclass'])['salary'].value_counts(normalize=True)
#sex

data.groupby(['sex'])['salary'].value_counts(normalize=True)
X = data.loc[:,['age','sex']]



# '''education', 

#                  'marital-status',

#                  'race',

#                  'sex',

#                  'workclass'''
X.head()
y = data.iloc[:,-1]

y.size

y.head()
#encoding male and female

X['sex'].replace(to_replace=[' Male',' Female'], value=[0,1],inplace=True)
X.head()
# '''education', 

#                  'marital-status',

#                  'race',

#                  'sex',

#                  'workclass'''

X = pd.concat([X,pd.get_dummies(data['education'])], axis=1)
X = pd.concat([X,pd.get_dummies(data['marital-status'])], axis=1)

X = pd.concat([X,pd.get_dummies(data['workclass'])], axis=1)
pd.set_option('display.max_columns', None)

X.head()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)
X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
y_train.head()
# Modeling

from sklearn.neighbors import KNeighborsClassifier

# Best k

Ks=15

mean_acc=np.zeros((Ks-1))

std_acc=np.zeros((Ks-1))

ConfustionMx=[];

for n in range(1,Ks):

    

    #Train Model and Predict  

    kNN_model = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)

    yhat = kNN_model.predict(X_test)

    

    

    mean_acc[n-1]=np.mean(yhat==y_test);

    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

k = mean_acc.argmax()+1

k
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

kNN_model
#generating the predicted y

knn_yhat = kNN_model.predict(X_test)
from sklearn.metrics import jaccard_similarity_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss

print("KNN Jaccard index: %.2f" % jaccard_similarity_score(y_test, knn_yhat))

print("KNN F1-score: %.2f" % f1_score(y_test, knn_yhat, average='weighted') )