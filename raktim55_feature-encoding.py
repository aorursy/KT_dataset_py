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
df_train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

df_test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')

print(df_train.shape, df_test.shape)

def description(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.iloc[0].values

    summary['Second Value'] = df.iloc[1].values

    summary['Third Value'] = df.iloc[2].values

    return summary

description(df_train)
description(df_test)
def replace_nan(data):

    for column in data.columns:

        if data[column].isna().sum() > 0:

            data[column] = data[column].fillna(data[column].mode()[0])





replace_nan(df_train)

replace_nan(df_test)
df_train.isna().sum()
df_test.isna().sum()
df_train_CE=df_train.copy()

columns=['day','month']

for col in columns:

    df_train_CE[col+'_sin']=np.sin((2*np.pi*df_train_CE[col])/max(df_train_CE[col]))

    df_train_CE[col+'_cos']=np.cos((2*np.pi*df_train_CE[col])/max(df_train_CE[col]))

df_train_CE=df_train_CE.drop(columns,axis=1)

df_train=df_train_CE

df_train.head()
df_train.shape
df_test_CE=df_test.copy()

columns=['day','month']

for col in columns:

    df_test_CE[col+'_sin']=np.sin((2*np.pi*df_test_CE[col])/max(df_test_CE[col]))

    df_test_CE[col+'_cos']=np.cos((2*np.pi*df_test_CE[col])/max(df_test_CE[col]))

df_test_CE=df_test_CE.drop(columns,axis=1)

df_test=df_test_CE

df_test.head()
df_test.shape
df_train=pd.get_dummies(df_train, prefix=['Color', 'Shape', 'Animal', 'Country', 'Musical Instrument', ], columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])

df_train
df_test=pd.get_dummies(df_test, prefix=['Color', 'Shape', 'Animal', 'Country', 'Musical Instrument', ], columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])

df_test
map_ord1 = {'Novice':1,

           'Contributor':2,

           'Expert':3,

           'Master':4,

           'Grandmaster':5}



map_ord2 = {'Freezing':1, 

            'Cold':2, 

            'Warm':3, 

            'Hot':4, 

            'Boiling Hot':5, 

            'Lava Hot':6}



map_bin3 = {'T':1,

           'F':0}



map_bin4 = {'Y':1,

           'N':0}



df_train.ord_1 = df_train.ord_1.map(map_ord1)

df_test.ord_1 = df_test.ord_1.map(map_ord1)



df_train.ord_2 = df_train.ord_2.map(map_ord2)

df_test.ord_2 = df_test.ord_2.map(map_ord2)



df_train.bin_3 = df_train.bin_3.map(map_bin3)

df_test.bin_3 = df_test.bin_3.map(map_bin3)



df_train.bin_4 = df_train.bin_4.map(map_bin4)

df_test.bin_4 = df_test.bin_4.map(map_bin4)
nom_hc = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
for col in nom_hc:

    df_train[f'hash_{col}'] = df_train[col].apply( lambda x: hash(str(x)) % 5000)

    df_test[f'hash_{col}'] = df_test[col].apply( lambda x: hash(str(x)) % 5000)
nom_hashed = ['hash_nom_5', 'hash_nom_6', 'hash_nom_7', 'hash_nom_8',

            'hash_nom_9']
df_train = df_train.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1)

df_test = df_test.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1)
description(df_train)
description(df_test)
from sklearn.preprocessing import LabelEncoder
df_train['ord_3_LE'] = LabelEncoder().fit_transform(df_train.ord_3)

df_train['ord_4_LE'] = LabelEncoder().fit_transform(df_train.ord_4)

df_train['ord_5_LE'] = LabelEncoder().fit_transform(df_train.ord_5)



df_train = df_train.drop(['ord_3', 'ord_4', 'ord_5'], axis=1)



description(df_train)
df_test['ord_3_LE'] = LabelEncoder().fit_transform(df_test.ord_3)

df_test['ord_4_LE'] = LabelEncoder().fit_transform(df_test.ord_4)

df_test['ord_5_LE'] = LabelEncoder().fit_transform(df_test.ord_5)



df_test = df_test.drop(['ord_3', 'ord_4', 'ord_5'], axis=1)



description(df_test)