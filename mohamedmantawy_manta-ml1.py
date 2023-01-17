# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read csv file

data = pd.read_csv('../input/train.csv')
data
#gets first 5 rows ...default is 5, can change it by adding number n in (n)

data.head()
#gets description of each table

data.describe()
#gets info about table ( , null/non-null , type of column)

data.info()
#gets true in the cell of the null value

data.isna()
#gets the sum of null values in each column

data.isna().sum()
#gets correlation between each column

data.corr()
#plots two columns between each other

plt.scatter(data['Cover_Type'],data['Elevation'])
# getting column "Cover_Type" into dataframe labels 

labels=data['Cover_Type']
#train["Cover_type"].value_counts()
# 

data=data.drop(['Cover_Type'],axis=1)

data=data.drop(['Id'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)
from sklearn import tree



from sklearn.ensemble import ExtraTreesClassifier

#clf = tree.DecisionTreeClassifier()

#n_estimators = [500 , 550 ,650 ,700 ,750 ,800 ,850 ,950]

clf = ExtraTreesClassifier(n_estimators =500, random_state=0)

clf = clf.fit(X_train, y_train)
clf.score(X_train,y_train)

clf.score(X_test,y_test)
