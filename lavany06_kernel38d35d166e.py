# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
df.head()
df.describe()
df.info()
df.isnull().any()
sns.set_style('whitegrid')

df['age'].hist(bins=30)

plt.xlabel('Age')
sns.jointplot(x='age',y='cigsPerDay', data = df, kind='kde')
sns.jointplot(x='cigsPerDay', y='heartRate', data=df)
sns.pairplot(df)
df.isnull().any()
df.fillna(df.mean(), inplace=True)
df.isnull().any()
#Target variable : 'TenYearCHD'



print(df.groupby('TenYearCHD').size())
X = df.drop(columns=['TenYearCHD'])
X.head()
y = df['TenYearCHD']
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))