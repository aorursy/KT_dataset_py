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
df_train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

df_test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
df_train.drop("id",axis = 1, inplace = True)

df_test.drop("id",axis = 1, inplace = True)
df_train.columns
df_train_bfeatures = df_train.iloc[:,0:4]

df_test_bfeatures = df_test.iloc[:,0:4]
df_train_bfeatures
df_train_nomfeatures = df_train.iloc[:,5:15]

df_test_nomfeatures = df_test.iloc[:,5:15]
df_train_nomfeatures.nunique()
df_train_nomfeatures
df_train_ordfeatures = df_train.iloc[:,16:21]

df_test_ordfeatures = df_test.iloc[:,16:21]
df_train
y = df_train.target
df_train.drop("target" ,axis = 1, inplace = True)
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

for cols in df_train_nomfeatures:

    df_train_nomfeatures[cols] = lb.fit_transform(df_train_nomfeatures[cols])

    df_test_nomfeatures[cols] = lb.fit_transform(df_test_nomfeatures[cols])
df_train_ordfeatures.nunique()
df_train_nomfeatures.head(50)
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

for cols in df_train_nomfeatures:

    df_train_nomfeatures[cols] = lb.fit_transform(df_train_nomfeatures[cols])

    df_test_nomfeatures[cols] = lb.fit_transform(df_test_nomfeatures[cols])
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

for cols in df_train_ordfeatures:

    df_train_ordfeatures[cols] = lb.fit_transform(df_train_ordfeatures[cols])

    df_test_ordfeatures[cols] = lb.fit_transform(df_test_ordfeatures[cols])
df_train['day'] = lb.fit_transform(df_train['day'])
df_train['month'] = lb.fit_transform(df_train['month'])
df_train.columns
df_train.drop(['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2',

       'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0',

       'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5',],axis = 1, inplace = True)
X = pd.concat([df_train_bfeatures,df_train_nomfeatures,df_train_ordfeatures],axis = 1,sort = False)
X
X = pd.concat([X,df_train['day'],df_train['month']],axis = 1, sort = False)
X
y
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
X