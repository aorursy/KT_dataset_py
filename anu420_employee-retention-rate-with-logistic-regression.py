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
df = pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')
df.head()
#check null value
df.isna().sum()
df.groupby('left').mean()
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
subdf.head()
salary_dummies = pd.get_dummies(subdf.salary, prefix = 'salary')
salary_dummies.head()

df_sub = pd.concat([subdf, salary_dummies], axis= 'columns')

df_sub = df_sub.drop('salary', axis = 'columns')
df_sub.head()
X = df_sub
y = df.left
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_train, y_train)
