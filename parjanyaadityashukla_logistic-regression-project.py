import pandas as pd 

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
ad_data = pd.read_csv('../input/advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()
plt.hist(ad_data['Age'])
sns.jointplot(x='Age',y='Area Income',data=ad_data)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde')
sns.jointplot(x='Daily Internet Usage',y='Daily Time Spent on Site',data=ad_data,)
sns.pairplot(data=ad_data,hue='Clicked on Ad')
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',

       'Daily Internet Usage','Male']]

ad_data.columns
y = ad_data['Clicked on Ad']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))