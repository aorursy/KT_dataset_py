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
df=pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')

df
df.head()
df.info()
df.describe()
pd.crosstab(df.salary,df.left).plot(kind='bar')
pd.crosstab(df.Department,df.left).plot(kind='bar')
df.groupby('left').mean()
import matplotlib.pyplot as plt

pd.crosstab(df.average_montly_hours,df.left).plot(kind='bar')
pd.crosstab(df.promotion_last_5years,df.left).plot(kind='bar')
dummies=pd.get_dummies(df.salary)

dummies
merged=pd.concat([df,dummies],axis='columns')

merged.head()
final=merged.drop(['salary','high'],axis='columns')

final.head()
dummies_1=pd.get_dummies(df.Department)

dummies_1.head()
merged_1=pd.concat([final,dummies_1],axis='columns')

merged_1.head()
merged_1.columns

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

merged_1['Department']=le.fit_transform(merged_1['Department'])
merged_1.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(merged_1.drop('left',axis='columns'),merged_1.left,test_size=0.2)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)