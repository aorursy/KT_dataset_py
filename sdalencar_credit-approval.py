import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]

df = pd.read_csv('/kaggle/input/creditscreening/credit-screening.data', sep=',',names = nomes, na_values = '?')

df.head()
df.describe()
df.isnull().sum()

df['A1']
catFaltante = ['A1','A4','A5','A6','A7']

for i in catFaltante:

    df[i].fillna(value=df[i].mode(), inplace = True)
contFaltante = ['A2','A14']

for i in contFaltante:

    df[i].fillna(value=df[i].mean(), inplace = True)

 

 
mudaCat = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']

for i in mudaCat:

     df[i].astype('category')

df['A16'] = df["A16"].astype('category')

df['A16'].cat.codes

df = pd.get_dummies( df, columns=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])

