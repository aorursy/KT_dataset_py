#Daniel Cruz @giordannocruz



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv',index_col=['Unnamed: 0'])

df2 = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
df1.head()
df1.info()
df1.describe(include = 'all').T
for col in df1:

    print(df1[col].value_counts(), '\n')
df1.isnull().sum()
df1.isna().sum()
furniture_mapper = {'not furnished': 0, 'furnished': 1}

animal_mapper= {'not acept': 0, 'acept': 1}

df1['furniture'].replace(furniture_mapper, inplace=True)

df1['animal'].replace(animal_mapper, inplace=True)

df1['floor'] = df1['floor'].replace('-',np.nan)

df1['floor'] = pd.to_numeric(df1['floor'])
df1.isnull().sum()
df1.describe(include='all').T
df1.floor.describe()
df1['floor'].replace(99,df1['floor'].mean(), inplace=True)

df1['floor'].fillna((math.floor(df1['floor'].mean())), inplace=True)
df1.floor.describe()
df1.isnull().sum()
df1.isna().sum()
df1.describe(include = 'all').T
df1.info()
df1["hoa"] = (df1["hoa"].str.strip("$R"))

df1["rent amount"] = (df1["rent amount"].str.strip("$R"))

df1["property tax"] = (df1["property tax"].str.strip("$R"))

df1["fire insurance"] = (df1["fire insurance"].str.strip("$R"))

df1["total"] = (df1["total"].str.strip("$R"))
df1.head()
df1["hoa"] = (df1["hoa"].str.replace(',', ''))

df1["rent amount"] = (df1["rent amount"].str.replace(',', ''))

df1["property tax"] = (df1["property tax"].str.replace(',', ''))

df1["fire insurance"] = (df1["fire insurance"].str.replace(',', ''))

df1["total"] = (df1["total"].str.replace(',', ''))

df1.head()
df1.info()
for i in df1.hoa:

    print(i)
df1['hoa'] = df1['hoa'].replace('Sem info',np.nan).replace('Incluso',np.nan)
df1['hoa'] = pd.to_numeric(df1['hoa'])
df1.isnull().sum()
df1.describe(include='all').T
df1['hoa'].replace(220000,df1['hoa'].mean(), inplace=True)

df1['hoa'].fillna((math.floor(df1['hoa'].mean())), inplace=True)
df1.info()
df1['property tax'] = df1['property tax'].replace('Sem info',np.nan).replace('Incluso',np.nan)
df1['property tax'] = pd.to_numeric(df1['property tax'])

df1.isnull().sum()
df1.describe(include='all').T
df1['property tax'].replace(366300,df1['property tax'].mean(), inplace=True)

df1['property tax'].fillna((math.floor(df1['property tax'].mean())), inplace=True)

df1.info()
#df1['property tax'] = pd.to_numeric(df1['rent amount'])

df1['rent amount'] = df1['rent amount'].astype('int64')

df1['property tax'] = df1['property tax'].astype('int64')

df1['fire insurance'] = df1['fire insurance'].astype('int64')

df1['total'] = df1['total'].astype('int64')

df1['floor'] = df1['floor'].astype('int64')

df1['hoa'] = df1['hoa'].astype('int64')

df1.info()
df1.head()
df1 = df1.drop(['total'], axis = 1) 
df1.head()
df1['total'] = df1['hoa']+df1['rent amount']+df1['property tax']+df1['fire insurance']
df1.head()
df2.head()
df2 = df2.rename(columns={'hoa (R$)': 'hoa', 'rent amount (R$)': 'rent amount',

                        'property tax (R$)':'property tax',

                        'fire insurance (R$)':'fire insurance','total (R$)':'total'})

df2.head()
df2.city.value_counts()
df2.info()
df2.isnull().sum()
df2.describe(include = 'all').T
df2['floor'] = df2['floor'].replace('-',np.nan)

df2['floor'] = pd.to_numeric(df2['floor'])

df2['parking spaces'] = df2['parking spaces'].astype('int64')

df2['furniture'].replace(furniture_mapper, inplace=True)

df2['animal'].replace(animal_mapper, inplace=True)

df2['hoa'] = df2['hoa'].replace('Sem info',np.nan).replace('Incluso',np.nan)

df2['hoa'] = pd.to_numeric(df2['hoa'])
df2.isnull().sum()
df2.floor.describe()
df2['floor'].replace(301,df2['floor'].mean(), inplace=True)

df2['floor'].fillna((math.floor(df2['floor'].mean())), inplace=True)

df2.isnull().sum()
df2['floor'] = df2['floor'].astype('int64')
df2.info()
corr_matrix  = df1.corr()

mask  = np.zeros_like(corr_matrix, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



# Correlation heatmap

fig, ax = plt.subplots(figsize = (15,9))



heatmap = sns.heatmap(corr_matrix,

                     mask = mask,

                     square = True,

                      cmap = "coolwarm",

                      cbar_kws = {"ticks": [-1,-0.5,0,0.5,1]},

                     vmin=-1,

                     vmax=1,

                     annot = True,

                     annot_kws = {"size": 10})



plt.show()