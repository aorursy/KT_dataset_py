# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#import pandas as pd
#import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.preprocessing  import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import r2_score 
from sklearn.preprocessing import Imputer


file = pd.read_csv('../input/train.csv')

df_train = pd.DataFrame(file) 

#print(df_train.head())
#print(df_train.describe())
print(df_train.info())

le = LabelEncoder()

columns = df_train.columns.values
#print(columns)

reg = linear_model.LinearRegression()

new_df = pd.DataFrame()

for column in columns:
    if df_train[column].dtype == np.int64 or df_train[column].dtype == np.float64:
        if (df_train[column].isnull().sum() == 0):
            new_df[column] = df_train[column]
        else:
            new_df[column] = df_train[column].fillna(df_train[column].mean())    
#    print(new_df.info())
    if df_train[column].dtype != np.int64 and df_train[column].dtype != np.float64:
        if(df_train[column].isnull().sum() == 0):
            new_df[column] = le.fit_transform(df_train[column])
        else:
            df_train[column].fillna('None',inplace=True)            
            new_df[column] = le.fit_transform(df_train[column])

print(new_df.describe())
print(new_df.info())

X = new_df.iloc[:,1:79]
Y = new_df.iloc[:,80]


print(X.head())
print(Y.head())
fit = reg.fit(X,Y)


Y_pred = reg.predict(X)

print(Y_pred)

print('r2  score = ' + str(r2_score(Y,Y_pred)))

print("---------------------------")       
print("---------------------------")        
print("Finished with trainning set")
print("---------------------------")
print("-Now dealing with test set-")        
print("---------------------------")
print("---------------------------")        
file2 = pd.read_csv('../input/test.csv')

df_test = pd.DataFrame(file2)

print(df_test.info())   

cols2 = df_test.columns.values

#print(cols2)  

df_ntest = pd.DataFrame()

for column in cols2:
    if df_test[column].dtype == np.int64 or df_test[column].dtype == np.float64:
        if (df_test[column].isnull().sum() == 0):
            df_ntest[column] = df_test[column]
        else:
            df_ntest[column] = df_test[column].fillna(df_test[column].mean())    
#    print(new_df.info())
    if df_test[column].dtype != np.int64 and df_test[column].dtype != np.float64:
        if(df_test[column].isnull().sum() == 0):
            df_ntest[column] = le.fit_transform(df_test[column])
        else:
            df_test[column].fillna('None',inplace=True)
            df_ntest[column] = le.fit_transform(df_test[column])

print(df_ntest.info())
print(df_ntest.describe())

#print(df_ntest.iloc[:,0])

X_test = df_ntest.iloc[:,1:79]
Y_test = reg.predict(X)
print(Y_test)
# Any results you write to the current directory are saved as output.
