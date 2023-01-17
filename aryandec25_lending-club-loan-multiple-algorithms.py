#Importing all the necessary library



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings 

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input/dataset/"))
#Let is convert or dataset into a DataFrame

loans=pd.read_csv('../input/dataset/loan_borowwer_data.csv')

loans.head()
loans.shape
loans.info()
loans.describe()
plt.figure(figsize=(10,6))

loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')

loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit.Policy=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(10,6))

loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='not.fully.paid=1')

loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')
purpose=list(loans['purpose'].unique())

purpose
plt.figure(figsize=(12,6))

sns.countplot(loans['purpose'])
loans['purpose'].value_counts()
sns.countplot(loans['not.fully.paid'])
plt.figure(figsize=(12,6))

sns.countplot(x=loans['purpose'],hue=loans['not.fully.paid'],palette='Set1')
corrmat=loans.corr()

plt.subplots(figsize=(10, 9))

sns.heatmap(corrmat, vmax=.8, square=True,cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10});
loans[['int.rate','inq.last.6mths','revol.util']].describe()
sns.jointplot(x='fico',y='int.rate',data=loans,color='green')
plt.figure(figsize=(11,7))

sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',

           col='not.fully.paid',palette='Accent_r')
loans.info()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
label_encoder=LabelEncoder()

loans['purpose']=label_encoder.fit_transform(loans['purpose'])
loans['purpose'].dtype
from sklearn.model_selection import train_test_split
X = loans.drop('not.fully.paid',axis=1)

y = loans['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,y_train)
predict_rfc = rfc.predict(X_test)
#Importing Logistic Regression

from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(X_train,y_train)
predict_Log=log.predict(X_test)
#importing a Support Vector machine

from sklearn.svm import SVC

svm=SVC(gamma='auto')

svm.fit(X_train,y_train)
predict_svm=svm.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print('\t\t CLASSIFICATION REPORT- Random Forest')

print(classification_report(y_test,predict_rfc))

print('\t  CLASSIFICATION REPORT- Logistic Regression')

print(classification_report(y_test,predict_Log))
sns.heatmap(confusion_matrix(y_test,predict_rfc),annot=True,fmt='')

plt.title('Confusion Matrix-Random Forest')
sns.heatmap(confusion_matrix(y_test,predict_Log),annot=True,fmt='')

plt.title('Confusion Matrix-Logistic Regression')
print('Accuracy Score of Random Forest Classifier: {:.2f}'.format(accuracy_score(y_test,predict_rfc)))

print('Accuracy Score of Logistic Regression Classifier: {:.2f}'.format(accuracy_score(y_test,predict_Log)))