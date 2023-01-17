# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Crime_Data.csv')

df.shape
df.head()
missing_val_count_by_column = (df.isnull().sum())

print('There is the amount of not available data in each column:\n',missing_val_count_by_column[missing_val_count_by_column > 0])

print('\nTop 10 crime descriptions are:\n',df['Primary Offense Description'].value_counts().head(10))

print('\nUnique categorical variables are\n:',df['Precinct'].unique())





df=df.mask(df.eq('UNKNOWN')).dropna(axis=0)

df=df.mask(df.eq('nan')).dropna(axis=0)

base=df.loc[df['Primary Offense Description'].isin(['THEFT-CARPROWL','THEFT-SHOPLIFT','THEFT-OTH', 'VEH-THEFT-AUTO',

       'BURGLARY-FORCE-RES', 'THEFT-BUILDING', 'BURGLARY-NOFORCE-RES',

       'TRESPASS', 'BURGLARY-FORCE-NONRES', 'DUI-LIQUOR'])]

tab=pd.crosstab(df.Precinct, base['Primary Offense Description'])

tab
tab.plot.bar(stacked=True,figsize=(16,10),grid=100)
def side(where):

    qq=df.loc[df['Precinct'].isin([where])]

    tab2=pd.crosstab(base['Primary Offense Description'], qq.Precinct)

    return tab2.plot.bar(stacked=True,figsize=(12,8),grid=100, rot=30)



sides=df.Precinct.unique()

sides

for sd in sides:

    function=side(sd)

    print(function)