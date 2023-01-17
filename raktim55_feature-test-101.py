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
df_train_AF = df_train[['nom_0', 'month', 'nom_5', 'day', 'ord_4', 'ord_2']]

df_test_AF = df_test[['nom_0', 'month', 'nom_5', 'day', 'ord_4', 'ord_2']]

df_train_AF.head()
my_AD=pd.concat([df_train_AF,df_test_AF], sort=False)

my_AD.shape
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

description(my_AD)
my_AD_CE=my_AD.copy()

columns=['day','month']

for col in columns:

    my_AD_CE[col+'_sin']=np.sin((2*np.pi*my_AD_CE[col])/max(my_AD_CE[col]))

    my_AD_CE[col+'_cos']=np.cos((2*np.pi*my_AD_CE[col])/max(my_AD_CE[col]))

my_AD_CE=my_AD_CE.drop(columns,axis=1)

my_AD=my_AD_CE

my_AD.head()
my_AD.shape
import missingno as msno
msno.matrix(my_AD)
msno.matrix(my_AD, sort='ascending')
my_AD=pd.get_dummies(my_AD, prefix=['Color'], columns=['nom_0'])

my_AD
map_ord2 = {'Freezing':1, 

            'Cold':2, 

            'Warm':3, 

            'Hot':4, 

            'Boiling Hot':5, 

            'Lava Hot':6}

my_AD.ord_2 = my_AD.ord_2.map(map_ord2)

my_AD
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['ord_4'])

df_bin = encoder.fit_transform(my_AD['ord_4'])

my_AD = pd.concat([my_AD,df_bin], axis=1)

my_AD
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['nom_5'])

df_bin = encoder.fit_transform(my_AD['nom_5'])

my_AD = pd.concat([my_AD,df_bin], axis=1)

my_AD