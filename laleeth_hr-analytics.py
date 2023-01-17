# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')
df.head(10)
df['salary'].value_counts()
df['Department'].value_counts()
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_new = df.drop(['Department','salary'],axis=1)
df_new.head()
df_left = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

df_left.head()
salary_dummies = pd.get_dummies(df_left.salary, prefix="salary")
df_salary_dummies = pd.concat([df_left,salary_dummies],axis='columns')
df_salary_dummies.head()
df_salary_dummies.drop('salary',axis='columns',inplace=True)

df_salary_dummies.head()
X = df_salary_dummies

y = df.left
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)
model = model.fit(X_train,y_train)

model
print(model.score(X_test,y_test))