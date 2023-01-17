#Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/data.csv", sep=',')

df
X = df.iloc[:,[0,1,2]].values

y = df.iloc[:,3].values
X
y
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")

imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

X
#independent variables

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



ct = ColumnTransformer(transformers=[('endoer', OneHotEncoder(), [0])], remainder='passthrough')

X = ct.fit_transform(X)

X = np.array(X)

X
#dependent variables

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

y
from sklearn.model_selection import train_test_split  #(for python2)

#from sklearn.model_selection import train_test_split  (for python3)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

print('X_train.shape: ', X_train.shape)

print('X_test.shape:  ', X_test.shape)

print('y_train.shape: ', y_train.shape)

print('y_test.shape:  ', y_test.shape)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
X_train
X_test