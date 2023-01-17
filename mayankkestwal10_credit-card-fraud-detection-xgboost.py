# Importing necessary libraries

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Loading data

data = pd.read_csv('../input/creditcard.csv')
# View DataFrame

data.head()
data.shape
data.columns
data.info()
data.describe()  #statistical inference
# Visualising every feature

data.hist(figsize=(20,20))

plt.show()
# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]

Valid = data[data['Class'] == 0]



outlier_fraction = len(Fraud)/(len(Valid))

print(outlier_fraction)



print('Fraud Cases : {}'.format(len(Fraud)))

print('Valid Cases : {}'.format(len(Valid)))
# Correlation

corr = data.corr()

figure = plt.figure(figsize=(12,10))

sns.heatmap(corr)
# Splitting data

x = data.iloc[:,:-1].values

y = data.iloc[:,-1].values



from sklearn.model_selection import train_test_split

xtr,xtest,ytr,ytest = train_test_split(x,y,test_size=0.3,random_state=0)
xtr.shape,ytr.shape
xtest.shape,ytest.shape
from xgboost import XGBClassifier

xg = XGBClassifier(random_state=0)

xg.fit(xtr,ytr)

xg.score(xtr,ytr)
pred = xg.predict(xtest)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(pred,ytest)

cm
from sklearn.metrics import accuracy_score

accuracy_score(pred,ytest)