import sys

import logging



import numpy as np

import scipy as sp

import sklearn

import pandas as pd

import statsmodels.api as sm

from statsmodels.formula.api import ols



import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import seaborn as sn

sn.set_context("poster")

sn.set(rc={'figure.figsize': (16, 9.)})

sn.set_style("whitegrid")



import pandas as pd

pd.set_option("display.max_rows", 120)

pd.set_option("display.max_columns", 120)



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/Iris.csv')
data.head()
data.drop(['Id'],inplace = True, axis = 1)
data.shape
data.head()
sn.pairplot(data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])
data.describe()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sn.violinplot( x=data["Species"], y=data["SepalLengthCm"])

plt.subplot(2,2,2)

sn.violinplot( x=data["Species"], y=data["SepalWidthCm"])

plt.subplot(2,2,3)

sn.violinplot( x=data["Species"], y=data["PetalLengthCm"])

plt.subplot(2,2,4)

sn.violinplot( x=data["Species"], y=data["PetalWidthCm"])
plt.figure(figsize=(7,4)) 

sn.heatmap(data.corr(),annot=True) 

plt.show()
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.25)

X_train = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

X_test = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_train = train['Species']

y_test = test['Species']
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=3) 

model.fit(X_train,y_train)

prediction_train=model.predict(X_train)

prediction_test=model.predict(X_test)

print('The accuracy of the KNN on train is',metrics.accuracy_score(prediction_train,y_train))

print('The accuracy of the KNN on test is',metrics.accuracy_score(prediction_test,y_test))
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(multi_class='auto') 

model.fit(X_train,y_train)

prediction_train=model.predict(X_train)

prediction_test=model.predict(X_test)

print('The accuracy of the Logistic on train is',metrics.accuracy_score(prediction_train,y_train))

print('The accuracy of the Logistic on test is',metrics.accuracy_score(prediction_test,y_test))
for i in [0.001,0.01,0.1,1,10,100,1000]:

    model=LogisticRegression(multi_class='auto',C=i) 

    model.fit(X_train,y_train)

    prediction_train=model.predict(X_train)

    prediction_test=model.predict(X_test)

    print('C is: {:.3f}'.format(i))

    print('The accuracy of the Logistic on train is',metrics.accuracy_score(prediction_train,y_train))

    print('The accuracy of the Logistic on test is',metrics.accuracy_score(prediction_test,y_test))

    print('Coefficients chage due to regularization hyperparameter:\n',model.coef_)

    print('')