import seaborn as sns 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
df.head()
df.describe()
df.corr()
sns.heatmap(df.corr())
sns.pairplot(df,hue='species')
df.info()
print(df.isnull().sum())
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
x = df.drop(['species'],axis=1)

y = df['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=np.random)
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(accuracy_score(y_test, predictions))