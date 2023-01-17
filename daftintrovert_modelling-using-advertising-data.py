import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
file = '../input/Advertising.csv'

data = pd.read_csv(file)
data.head()
data.drop(['Unnamed: 0'],axis = 1,inplace = True)
data.info()
data.describe()
sns.heatmap(data.corr(),cmap = 'magma',lw = .7,linecolor = 'lime',alpha = 0.8,annot = True)
sns.distplot(data['sales'],hist = True)
sns.jointplot(x = 'sales',y = 'TV',data = data,kind = 'kde',color = 'red')
sns.jointplot(x = 'sales',y = 'newspaper',data = data,kind = 'kde',color = 'green',hist = True)
sns.jointplot(x = 'sales',y = 'radio',data = data,kind = 'kde',color = 'gold',hist = True)
sns.pairplot(data,height = 4)
X = data.drop(['sales'],axis = 1)

y = data['sales']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logmodel =  LogisticRegression()
from sklearn import preprocessing

from sklearn import utils



lab_enc = preprocessing.LabelEncoder()

encoded = lab_enc.fit_transform(y_train)
lab_enc = preprocessing.LabelEncoder()

encoded2 = lab_enc.fit_transform(y_test)
logmodel.fit(X_train,encoded)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(encoded2,predictions))

print(confusion_matrix)