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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

DataFrame = pd.read_csv("../input/salaries.csv")
DataFrame
DataFrame.shape
DataFrame.size
DataFrame.head()
print(DataFrame.columns)

print("Null values ")

print(DataFrame.isna().sum())
X= DataFrame[['company','job','degree']]

y= DataFrame['salary_more_then_100k']
print(X)

print(y)
company_le = LabelEncoder()

job_le = LabelEncoder()

degree_le = LabelEncoder()
X['company_le'] = company_le.fit_transform(X['company'])

X['job_le'] = job_le.fit_transform(X['job'])

X['degree_le'] = degree_le.fit_transform(X['degree'])

X
X.drop(['company','job','degree'],axis='columns',inplace=True)
X
from sklearn.model_selection import train_test_split

X_Train, X_Test, y_Train, y_Test  = train_test_split(X,y,test_size=0.2,random_state=10)
print(X_Train)
print(X_Test)
model= DecisionTreeClassifier()
model.fit(X,y)
model.predict([[2,2,1]])
model.score(X,y)