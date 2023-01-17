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
df=pd.read_csv('/kaggle/input/insurance-claim/insurance_claims.csv')
df
df.columns
df.isnull().sum()
df.info()
df['policy_state'].unique()

import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x=df['fraud_reported'],hue=df['policy_state'],data=df)
sns.countplot(x=df['fraud_reported'],hue='insured_sex',data=df)
sns.countplot(x=df['fraud_reported'],hue=df['insured_education_level'],data=df)
sns.countplot(x=df['fraud_reported'],hue='insured_occupation',data=df)
sns.countplot(x=df['fraud_reported'],hue='witnesses',data=df)
df['police_report_available']=df['police_report_available'].replace({'?':np.nan})
df['police_report_available']=df['police_report_available'].fillna(method='ffill')
df['collision_type']=df['collision_type'].replace({'?':np.nan})
df['collision_type']=df['collision_type'].fillna(method='ffill')
df['property_damage']=df['property_damage'].replace({'?':np.nan})
df['property_damage']=df['property_damage'].fillna(method='ffill')
sns.countplot(x=df['fraud_reported'],hue='police_report_available',data=df)
#lets drop the unnecessary feature 

df=df.drop(['months_as_customer','policy_number','policy_bind_date','policy_csl','auto_year','auto_model','insured_hobbies','insured_zip'],axis=1)
#now deal with the categorical features

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
df
y=df['fraud_reported']

x=df.drop(['fraud_reported'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier()

gbc.fit(x_train,y_train)

pred_3=gbc.predict(x_test)

score_3=accuracy_score(y_test,pred_3)
score_3
from xgboost import XGBClassifier

xgb= XGBClassifier()

xgb.fit(x_train,y_train)

pred_4=xgb.predict(x_test)

score_4=accuracy_score(y_test,pred_4)
score_4