# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/loan-predication'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data = pd.read_csv('/kaggle/input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')
raw_data.shape
raw_data.head()
raw_data.isnull().sum()
raw_data.info()
import seaborn as sns

sns.countplot('Loan_Status',hue='Gender',data=raw_data)
raw_data['Gender'] = raw_data['Gender'].fillna('Male')
sns.countplot('Loan_Status',hue='Married',data=raw_data)
raw_data['Married'] = raw_data['Married'].fillna('Yes')
sns.countplot('Loan_Status',hue='Dependents',data=raw_data)
raw_data['Dependents'] = raw_data['Dependents'].fillna('0')
sns.countplot('Loan_Status',hue='Self_Employed',data=raw_data)
raw_data['Self_Employed'] = raw_data['Self_Employed'].fillna('Yes')
sns.distplot(raw_data['LoanAmount'])
sns.scatterplot(raw_data['LoanAmount'],y=np.arange(0,614))
mean=raw_data[raw_data['LoanAmount']<=400]['LoanAmount'].mean()

raw_data['LoanAmount'].fillna(mean,inplace=True)

sns.scatterplot(raw_data['Loan_Amount_Term'],y=np.arange(0,614))
raw_data['Loan_Amount_Term'].fillna(raw_data['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)

raw_data['Credit_History'].unique()
sns.countplot('Loan_Status',hue='Credit_History',data=raw_data)
raw_data['Credit_History'].fillna(raw_data['Credit_History'].value_counts().idxmax(), inplace=True)
raw_data.isnull().sum()
raw_data.head()

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

df = raw_data.copy()
print(df['Gender'].unique())

print(df['Married'].unique())

print(df['Education'].unique())

print(df['Self_Employed'].unique())

print(df['Property_Area'].unique())
df['Gender'] = lb.fit_transform(df['Gender'])

df['Married'] = lb.fit_transform(df['Married'])

df['Education'] = lb.fit_transform(df['Education'])

df['Self_Employed'] = lb.fit_transform(df['Self_Employed'])

df['Property_Area'] = lb.fit_transform(df['Property_Area'])

df['Loan_Status'] = lb.fit_transform(df['Loan_Status'])

df['Dependents'] = lb.fit_transform(df['Dependents'])
print(df['Gender'].unique())

print(df['Married'].unique())

print(df['Education'].unique())

print(df['Self_Employed'].unique())

print(df['Property_Area'].unique())

print(df['Loan_Status'].unique())

print(df['Dependents'].unique())
df.head()
df.columns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

std = StandardScaler()
unscaled_features = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]

scaled_features = std.fit_transform(unscaled_features)

df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']] = scaled_features
df.head()
X = df.drop(['Loan_ID','Loan_Status'],axis=1)

y = df['Loan_Status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

lgt = LogisticRegression()

lgt.fit(X_train,y_train)

predict = lgt.predict(X_test)

print(classification_report(y_test,predict))
#Using Xgboost Classifier

import xgboost as xgb

xgb = xgb.XGBClassifier()

xgb.fit(X_train,y_train)

xgb.score(X_test,y_test)

# Using catboost Classifier

import catboost as cbg

cb = cbg.CatBoostClassifier()

cb.fit(X_train,y_train)

cb.score(X_test,y_test)

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(base_estimator=lgt)

ada.fit(X_train,y_train)

ada.score(X_test,y_test)