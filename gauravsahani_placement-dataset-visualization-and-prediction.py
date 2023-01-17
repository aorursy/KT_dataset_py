import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df
df.isnull().sum()
df['salary']=df['salary'].fillna(df['salary'].mode()[0])

df.isnull().sum()
sns.set_style('whitegrid')

sns.countplot(x = 'degree_t', data = df)

plt.title('Degree Comparison')

df['degree_t'].value_counts()
sns.set_style('darkgrid')

sns.countplot(x = 'gender', hue = 'status', data = df)

plt.title('No of students placed amongst gender')
sns.set_style('darkgrid')

sns.countplot(x = 'specialisation', hue = 'status', data = df)

plt.title('No of students placed amongst gender')
sns.catplot(x="status", y="ssc_p", data=df,kind="swarm",hue='gender')

sns.catplot(x="status", y="hsc_p", data=df,kind="swarm",hue='gender')

sns.catplot(x="status", y="degree_p", data=df,kind="swarm",hue='gender')
sns.catplot(x="workex", kind="count",hue ='gender', data=df, col='status');
sns.jointplot(x=df['mba_p'], y=df['salary']);
df['gender'].replace(to_replace='M', value=1, inplace=True)

df['gender'].replace(to_replace='F', value=0, inplace=True)



df['ssc_b'].replace(to_replace='Central', value=1, inplace=True)

df['ssc_b'].replace(to_replace='Others', value=0, inplace=True)



df['hsc_b'].replace(to_replace='Central', value=1, inplace=True)

df['hsc_b'].replace(to_replace='Others', value=0, inplace=True)



df['hsc_s'].replace(to_replace='Science', value=1, inplace=True)

df['hsc_s'].replace(to_replace='Commerce', value=2, inplace=True)

df['hsc_s'].replace(to_replace='Arts', value=3, inplace=True)



df['workex'].replace(to_replace='Yes', value=1, inplace=True)

df['workex'].replace(to_replace='No', value=0, inplace=True)



df['specialisation'].replace(to_replace='Mkt&Fin', value=1, inplace=True)

df['specialisation'].replace(to_replace='Mkt&HR', value=0, inplace=True)
df.degree_t.unique()
df['degree_t'].replace(to_replace='Sci&Tech', value=1, inplace=True)

df['degree_t'].replace(to_replace='Comm&Mgmt', value=2, inplace=True)

df['degree_t'].replace(to_replace='Others', value=2, inplace=True)



df['status'].replace(to_replace='Placed', value=1, inplace=True)

df['status'].replace(to_replace='Not Placed', value=0, inplace=True)



df.drop(['sl_no'],axis=1,inplace=True)

df.drop(['salary'],axis=1,inplace=True)
df.head(10)
df.dtypes
X = df[['gender', 'ssc_p','ssc_b', 'hsc_p','hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex','etest_p', 'specialisation', 'mba_p',]]

y = df['status']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix
dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)
pred1=dt.predict(X_test)

accuracy_score(pred1,y_test)
confusion_matrix(pred1,y_test)
rf=RandomForestClassifier()

rf.fit(X_train,y_train)
pred2=rf.predict(X_test)

accuracy_score(pred2,y_test)
confusion_matrix(pred2,y_test)
import xgboost as xgb

xgb=xgb.XGBClassifier()

xgb.fit(X_train,y_train)
pred3=xgb.predict(X_test)

accuracy_score(pred3,y_test)
confusion_matrix(pred3,y_test)