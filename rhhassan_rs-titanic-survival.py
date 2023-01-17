import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#import pandas_profiling 
#data = pd.read_csv("../input/titanic/train.csv", index_col="PassengerId")

data = pd.read_csv("../input/titanic/train.csv")

data2 = pd.read_csv("../input/titanic/test.csv")

data.shape
data.head()
data.describe()
data2.describe()

data2.columns
data.columns

X_train = data.drop(['PassengerId','Survived'],axis=1)

y_train = data.Survived

X_test = data2.drop(['PassengerId'],axis=1)
s = (data.dtypes == 'object')

catg_col = list(s[s].index)

print('Categorical : ',catg_col)
from sklearn.impute import SimpleImputer

im = SimpleImputer()

X_train = pd.DataFrame(im.fit_transform(X_train))

X_
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



label_X_train = X_train.copy()

label_X_test = X_test.copy()



for i in catg_col:

    label_X_train[i] = le.fit_transform(X_train[i])

    label_X_test[i] = le.transform(X_test[i])

profile = pandas_profiling.ProfileReport(data)

profile

data.index_col()