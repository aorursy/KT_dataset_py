import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
ad_data = pd.read_csv('advertising.csv')
ad_data.head()

ad_data.info()
ad_data.describe()
sns.distplot(ad_data['Age'],kde=False,bins=20)
sns.jointplot(x='Age',y='Area Income',data = ad_data)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data = ad_data,kind = 'kde')
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data = ad_data)
sns.pairplot(ad_data,hue = 'Clicked on Ad')
from sklearn.model_selection import train_test_split

ad_data.head(2)

ad_data['Ad Topic Line'].value_counts()
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
preditions = lr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,preditions))