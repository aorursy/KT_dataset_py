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
#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/Data.csv')
print(dataset)
X = dataset.iloc[:, :-1].values #.values -> np.ndarray type | else we get DataFrame obj
y = dataset.iloc[:, -1].values
print(X,'\n\n\n',y)
#DEALING WITH MISSING DATA
'''
if 
    missing data = 1% of total data (insignificant) then we can remove them 
else
    deal with them! replace by mean(mostly used), median or mode

'''
#Dealing with missing data using Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

#fit connects imputer to the dataset, 1st col excluded bc datatype = string -> can't find mean
imputer.fit(X[:, 1:3])

#transform -> returns df with replacement
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)
#encoding categorical data
'''
Categorical Data: Data with only certain values, here France, Germany, Spain -> 3 categories
Encoding -> string to number and tell that the values (0,1,2) are not related, they're just labels

ONE HOT ENCODING (for binary outcomes)
Value     France Germany Spain
France    1       0      0
Spain     0       0      1  
'''

#Encoding the independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

'''
ct args --
transformers: 1. kind of transformation , 2. type of transformation, 3. indexes of col to apply tf 
remainder: passthrough will keep all cols otherwise removes them and only keeps new cols resulted from encoding
'''
ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
print(X)
print(type(X))
#Encoding dependent variable using LabelEncoder
#LabelEncoder encodes in a single column taking values 0,1,2...
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
print(type(y))
#Splitting data into train and test set
#Original data -> broken into train and test. We know the outcomes of test data.
#Train used to train the data on and test data on which the trained model is tested, and (observed - acutal = error)
#by which accuracy is determined

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
#Feature Scaling : done after splitting, don't disturb the test data ONLY REQUIRED FOR SOME ML MODELS

'''
Standardisation :  (x - mean(x)) / stdev(x) -> results -3 to +3 generally |  StandardScaler
Normalisation   :  (x - min(x)) / (max(x) - min(x)) -> value b/w 0 and 1  |  MinMaxScaler

Which to use?
    Normalisation :  RECOMMENDED if NORMAL DISTRIBUTION for most features
    Standardisation : ALWAYS WORKS!

WHEN CALCULATING MEAN, DON'T INCLUDE TEST DATA BECAUSE IT'S SOMETHING NEW, UNEXPECTED
WHEN EVALUATING IRL WE CAN'T INCLUDE WHILE CALCULATING THE MEAN --> DND THE TEST DATA
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

'''
Apply scaler to dummy (Encoded) variables? 
-> OBVIOUSLY NOT! :P Will totally ruin the encoded values! Might improve performance but still. :o
'''
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])

#APPLY THE SAME SCALER TO X_TEST NOT A NEW SCALER BECAUSE (NEW SCALER -> NEW VALUES)
#fit: calculates, transform: changes values so, use only transform!
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)
