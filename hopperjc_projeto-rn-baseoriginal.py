# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import math

import statistics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/basern/base_train", sep="\t")

#df_train.head()
df_valid = pd.read_csv("../input/basern/base_valid", sep="\t")

#df_valid.head()
idadeMedia = df_train['IDADE'].mean()

idadeMedia = math.floor(idadeMedia)

df_train.update(df_train['IDADE'].fillna(idadeMedia))
#df_train.isnull().sum()
#df_valid.isnull().sum()
df_train.drop('REF_DATE', axis=1, inplace=True)

df_valid.drop('REF_DATE', axis=1, inplace=True)



df_train.drop('LATITUDE', axis=1, inplace=True)

df_valid.drop('LATITUDE', axis=1, inplace=True)



df_train.drop('LONGITUDE', axis=1, inplace=True)

df_valid.drop('LONGITUDE', axis=1, inplace=True)



df_train.drop('FLAG_OBITO', axis=1, inplace=True)

df_valid.drop('FLAG_OBITO', axis=1, inplace=True)
for i in range(0, len(df_train.columns)):

    if df_train.iloc[i].dtype == 'object':

        df_train.update(df_train[df_train.columns[i]].fillna('MD'))

    elif(df_train.iloc[i].dtype == 'int32'):

        mediaInt = df_train[df_train.columns[i]].mean()

        mediaInt = math.floor(mediaInt)

        df_train.update(df_train[df_train.columns[i]].fillna(mediaInt))

    else:

        df_train.update(df_train[df_train.columns[i]].fillna(statistics.mean(df_train.columns[i])))
for i in range(0, len(df_valid.columns)):

    if df_valid.iloc[i].dtype == 'object':

        df_valid.update(df_valid[df_valid.columns[i]].fillna('MD'))

    elif(df_valid.iloc[i].dtype == 'int32'):

        mediaInt = df_valid[df_valid.columns[i]].mean()

        mediaInt = math.floor(mediaInt)

        df_valid.update(df_valid[df_valid.columns[i]].fillna(mediaInt))

    else:

        df_valid.update(df_valid[df_valid.columns[i]].fillna(statistics.mean(df_valid.columns[i])))
X_train = df_train.iloc[:,0].values

Y_train = df_train.iloc[:,1:].values



X_valid = df_valid.iloc[:,0].values

Y_valid = df_valid.iloc[:,1:].values
for i in range(0, len(df_train.columns)):

    if (df_train.iloc[i].dtype == 'object'):

        df_train[df_train.columns[i]] = df_train[df_train.columns[i]].astype('category')        

df_train.dtypes
df_train.astype('category').cat.codes
LE = LabelEncoder()

for i in range(0, len(df_train.columns)):

    if(df_train.iloc[i].dtype == 'category'):

        df_train['code_', df_train.columns[i]] = LE.fit_transform(df_train[df_train.columns[i]])
for i in range(0, len(df_valid.columns)):

    if (df_valid.iloc[i].dtype == 'object'):

        df_valid[df_valid.columns[i]] = df_valid[df_valid.columns[i]].astype('category')        

df_valid.dtypes