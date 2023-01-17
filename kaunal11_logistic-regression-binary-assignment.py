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
df=pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')

df
left=df[df['left']==1]

left.mean()
retained=df[df['left']==0]

retained.mean()
pd.crosstab(df.salary,df.left).plot(kind='bar')

pd.crosstab(df.Department,df.left).plot(kind='bar')

subdf=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

subdf.head()
dummies=pd.get_dummies(df.salary)

merged=pd.concat([subdf,dummies],axis='columns')

merged.head()
X=merged.drop(['salary'],axis='columns')

X.head()
y=df.left

y.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model=LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)