# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
data.corr().Outcome.drop('Outcome').plot.bar()
sns.countplot(data['Pregnancies'],hue=data['Outcome'])
#we can see higher the no. of pregnancies higher is the chance that the patient is diabetic.
ratio=(data[data['Outcome']==1].groupby('Pregnancies').count()['Outcome'])/(data.groupby('Pregnancies').count()['Outcome'])
ratio.plot()
sns.distplot(data[data.Outcome==0]['Glucose'],bins=20)
sns.distplot(data[data.Outcome==1]['Glucose'],bins=20)
#Higher the Glucose level higher the chance of being diabetic.(Glucose level - 120 )
#We can see some outliers with a Glucose level of 0 and that's practically impossible. So we will drop them out
data.drop(data[data['Glucose']==0].index,inplace=True)
sns.distplot(data[data.Outcome==0]['BloodPressure'])
sns.distplot(data[data.Outcome==1]['BloodPressure'])
#Higher the Blood Pressure level higher the chance of being diabetic
#again we can see some outliers with zero blood pressure
data.drop(data[data['BloodPressure']==0].index,inplace=True)
sns.distplot(data[data.Outcome==0]['SkinThickness'],bins=20)
sns.distplot(data[data.Outcome==1]['SkinThickness'],bins=20)
# Diabetic condition has a very slight relation with skin thickness.
sns.distplot(data[data.Outcome==0]['Insulin'],bins=30)
sns.distplot(data[data.Outcome==1]['Insulin'],bins=30)
#Very high probability of diabetes if insulin levels are high
sns.distplot(data[data.Outcome==0]['BMI'])
sns.distplot(data[data.Outcome==1]['BMI'])
#BMI threshhold level = 30
sns.distplot(data[data.Outcome==0]['DiabetesPedigreeFunction'],bins=20)
sns.distplot(data[data.Outcome==1]['DiabetesPedigreeFunction'],bins=20)
#threshhold level = 0.5(approx)
sns.distplot(data[data.Outcome==0]['Age'])
sns.distplot(data[data.Outcome==1]['Age'])
#BMI threshhold level = 28(approx)
X=data.drop('Outcome',axis=1)
y=data['Outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(X_train,y_train)
pred=lg.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#We can see a 75% accuracy.
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
pred=dtc.predict(X_test)
print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))
# 67% accuracy 
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
pred=rfc.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#70% Accuracy.
models=[lg,dtc,rfc]
models
