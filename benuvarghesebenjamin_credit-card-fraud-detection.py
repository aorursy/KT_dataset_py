import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pylab import rcParams

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data.isnull().sum()
data.info()
data.describe().T.head()
data.shape
data.columns
fraud_cases=len(data[data['Class']==1])
print(' Number of Fraud Cases:',fraud_cases)
non_fraud_cases=len(data[data['Class']==0])
print('Number of Non Fraud Cases:',non_fraud_cases)
fraud=data[data['Class']==1]
genuine=data[data['Class']==0]
fraud.Amount.describe()
genuine.Amount.describe()
data.hist(figsize=(20,20),color='lime')

plt.show()
rcParams['figure.figsize'] = 16, 8

f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraud.Time, fraud.Amount)

ax1.set_title('Fraud')

ax2.scatter(genuine.Time, genuine.Amount)

ax2.set_title('Genuine')

plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()

plt.figure(figsize=(10,8))

corr=data.corr()

sns.heatmap(corr,cmap='BuPu')
from sklearn.model_selection import train_test_split
X=data.drop(['Class'],axis=1)
y=data['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
model=rfc.fit(X_train,y_train)
prediction=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
from sklearn.linear_model import LogisticRegression
X1=data.drop(['Class'],axis=1)
y1=data['Class']
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3,random_state=123)
lr=LogisticRegression()
model2=lr.fit(X1_train,y1_train)
prediction2=model2.predict(X1_test)
accuracy_score(y1_test,prediction2)
from sklearn.tree import DecisionTreeRegressor
X2=data.drop(['Class'],axis=1)
y2=data['Class']
dt=DecisionTreeRegressor()
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3,random_state=123)
model3=dt.fit(X2_train,y2_train)
prediction3=model3.predict(X2_test)
accuracy_score(y2_test,prediction3)