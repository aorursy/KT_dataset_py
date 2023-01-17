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
import pandas as pd

from matplotlib import pyplot as plt

df = pd.read_csv("/kaggle/input/hr-analytics/HR_comma_sep.csv")

df
left = df[df.left==1]

left.shape
retain = df[df.left == 0]

retain.shape
df.groupby('left').mean()
pd.crosstab(df.salary,df.left).plot(kind='bar')
pd.crosstab(df.Department,df.left).plot(kind='bar')


subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

subdf.head()
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.head()


df_with_dummies.drop('salary',axis='columns',inplace=True)

df_with_dummies.head()
X = df_with_dummies

X.head()
y = df.left


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)