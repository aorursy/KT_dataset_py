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
#for linear algebra and data processing 

import numpy as np 

import pandas as pd 
#for graphs

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/loandata/Loan payments data.csv')

df.head()
df.info()
sns.countplot(x='loan_status',data=df)
df['paid_off_time'].isnull().value_counts()
df['past_due_days'].isnull().value_counts()
df = df.drop('paid_off_time',axis=1)
df.columns
df['past_due_days'].fillna(value=0,inplace=True)
df.head()
g = sns.FacetGrid(data=df, col='Gender', row='education', hue='loan_status', sharex=False)

g = g.map(plt.scatter, 'age', 'past_due_days').add_legend(bbox_to_anchor=(1.2,0.5))

plt.tight_layout()
df.groupby('loan_status')['age'].describe()
df.groupby('loan_status')['education'].describe()
df.groupby('loan_status')['Gender'].describe()
dummies = pd.get_dummies(df['Gender'], drop_first=True)

df = df.drop('Gender', axis=1)

df = pd.concat([df,dummies], axis=1)
df.head()
df['education'].unique()
dummies_edu = pd.get_dummies(df['education'], drop_first=True)

df = df.drop('education', axis=1)

pd.concat([df, dummies_edu], axis=1)
df['effective_date_month'] = pd.DatetimeIndex(df['effective_date']).month

df['effective_date_year'] = pd.DatetimeIndex(df['effective_date']).year

df['due_date_month'] = pd.DatetimeIndex(df['due_date']).month

df['due_date_year'] = pd.DatetimeIndex(df['due_date']).year

df = df.drop(['effective_date', 'due_date'], axis=1)
df.head()
df.info()
df['effective_date_month'].unique()
df['effective_date_year'].unique()
df['due_date_month'].unique()
df['due_date_year'].unique()
df = df.drop(['effective_date_month', 'effective_date_year', 'due_date_month', 'due_date_year'], axis=1)

df.head()
plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), annot=True, cmap='viridis')

plt.tight_layout()
df = df.drop('Loan_ID', axis=1)

df.head()
from sklearn.model_selection import train_test_split
X = df.drop('loan_status', axis=1)

y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))