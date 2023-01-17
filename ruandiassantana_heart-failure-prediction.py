# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.info()
df.corr()['DEATH_EVENT'].sort_values()[:-1] 
sns.countplot(x='sex',data=df,hue='DEATH_EVENT')
df.corr()['DEATH_EVENT']['sex']
df.groupby('sex')['DEATH_EVENT'].mean()
df.groupby('sex')['DEATH_EVENT'].std()
df.groupby('DEATH_EVENT')['sex'].mean()
df.groupby('DEATH_EVENT')['sex'].std()
df['sex'].std()
df['sex'].mean()
df['DEATH_EVENT'].mean()
df['DEATH_EVENT'].std()
df.drop('sex',axis=1,inplace=True)
df.head()
sns.barplot(x='DEATH_EVENT',y='age',data=df)
df.groupby('DEATH_EVENT')['age'].mean()
df.groupby('DEATH_EVENT')['age'].std()
df.corr()['DEATH_EVENT']['age']
df['age'].mean()
df['age'].std()
df.corr()['DEATH_EVENT']['anaemia']
sns.countplot(x='anaemia',data=df,hue='DEATH_EVENT')
df.groupby('anaemia')['DEATH_EVENT'].mean()
df.groupby('anaemia')['DEATH_EVENT'].std()
df.groupby('DEATH_EVENT')['anaemia'].mean()
df.groupby('DEATH_EVENT')['anaemia'].std()
df.corr()['DEATH_EVENT']['anaemia']
df.head()
sns.distplot(df['creatinine_phosphokinase'],bins=50,kde=False)
df.groupby('DEATH_EVENT')['creatinine_phosphokinase'].mean()
df.groupby('DEATH_EVENT')['creatinine_phosphokinase'].std()
df['creatinine_phosphokinase'].mean()
df['creatinine_phosphokinase'].std()
df.corr()['creatinine_phosphokinase']['DEATH_EVENT']
sns.barplot(x='DEATH_EVENT',y='creatinine_phosphokinase',data=df)
df['diabetes']
sns.countplot(x='diabetes',data=df,hue='DEATH_EVENT')
df.groupby('diabetes')['DEATH_EVENT'].mean()
df.groupby('diabetes')['DEATH_EVENT'].std()
df.groupby('DEATH_EVENT')['diabetes'].mean()
df.groupby('DEATH_EVENT')['diabetes'].std()
df['diabetes'].std()
df.corr()['diabetes']['DEATH_EVENT']
df.drop('diabetes',axis=1,inplace=True)
df.head()
df.describe()['ejection_fraction']
sns.barplot(x='DEATH_EVENT',y='ejection_fraction',data=df)
sns.distplot(df['ejection_fraction'])
df.groupby('DEATH_EVENT')['ejection_fraction'].mean()
df.groupby('DEATH_EVENT')['ejection_fraction'].std()
df['ejection_fraction'].mean()
df['ejection_fraction'].std()
df.corr()['ejection_fraction']['DEATH_EVENT']
df.info()
df['high_blood_pressure']
sns.countplot(x='high_blood_pressure',data=df,hue='DEATH_EVENT')
df.groupby('high_blood_pressure')['DEATH_EVENT'].mean()
df.groupby('high_blood_pressure')['DEATH_EVENT'].std()
df.groupby('DEATH_EVENT')['high_blood_pressure'].mean()
df.groupby('DEATH_EVENT')['high_blood_pressure'].std()
df.corr()['high_blood_pressure']['DEATH_EVENT']
df.head()
sns.boxplot(x='DEATH_EVENT',y='platelets',data=df)
df.groupby('DEATH_EVENT')['platelets'].mean()
df.groupby('DEATH_EVENT')['platelets'].std()
df.corr()['platelets']['DEATH_EVENT']
df['platelets'].mean()
df['platelets'].std()
df['platelets'].max()
df['platelets'].min()
df.drop('platelets',axis=1,inplace=True)
df.info()
df['serum_creatinine'].nunique()
sns.distplot(df['serum_creatinine'])
df['serum_creatinine']
sns.barplot(x='DEATH_EVENT',y='serum_creatinine',data=df)
df.groupby('DEATH_EVENT')['serum_creatinine'].mean()
df.groupby('DEATH_EVENT')['serum_creatinine'].std()
df['serum_creatinine'].mean()
df['serum_creatinine'].std()
df.corr()['serum_creatinine']['DEATH_EVENT']
df.info()
df['serum_sodium']
sns.distplot(df['serum_sodium'])
sns.boxplot(x='DEATH_EVENT',y='serum_sodium',data=df)
df.groupby('DEATH_EVENT')['serum_sodium'].mean()
df.groupby('DEATH_EVENT')['serum_sodium'].std()
df.corr()['serum_sodium']['DEATH_EVENT']
df['serum_sodium'].mean()
df['serum_sodium'].std()
df.info()
df['smoking']
sns.countplot(x='smoking',data=df,hue='DEATH_EVENT')
df.groupby('smoking')['DEATH_EVENT'].mean()
df.groupby('smoking')['DEATH_EVENT'].std()
df.groupby('DEATH_EVENT')['smoking'].mean()
df.groupby('DEATH_EVENT')['smoking'].std()
df['smoking'].mean()
df['smoking'].std()
df.corr()['smoking']['DEATH_EVENT']
df.drop('smoking',axis=1,inplace=True)
df.head()
df['time']
df['time'].nunique()
sns.distplot(df['time'])
sns.barplot(x='DEATH_EVENT',y='time',data=df)
df.groupby('DEATH_EVENT')['time'].mean()
df.groupby('DEATH_EVENT')['time'].std()
df['time'].mean()
df['time'].std()
df.corr()['time']['DEATH_EVENT']
df.head()
df.isnull().sum()
from sklearn.model_selection import train_test_split
X = df.drop('DEATH_EVENT',axis=1)

y = df['DEATH_EVENT']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(verbose=7,random_state=23)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))