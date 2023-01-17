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
df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col=0)
df.head()
df.country
df['country']
df['country'][0]
df.iloc[0]
df.iloc[:, 0]
df.columns
df.iloc[0]
df.iloc[:, 1]
df.iloc[:3, 0]
df.iloc[:3, 1]
df.iloc[-5: ]
df.loc[0, 'country']
df.loc[1, 'description']
df.set_index('title')
#df.head()
df.country == 'Italy'
df.loc[df.country == 'France']
df.loc[(df.country == 'Italy') & (df.points >= 90)]

df.loc[(df.country == 'US') & (df.points >= 90)]
df['price']
df.price
df.price.mean()
df.loc[(df.country == 'Italy') | (df.points >= 90)]
df.loc[df.country.isin(['Italy', 'US'])]
df.loc[(df.country.isin(['Italy','US'])) & (df.price >= 40)]
df.loc[df.price.notnull()]
df.loc[df.designation.notnull()]
df['Acitic'] ='EveryOne'
df['Acitic']
df.describe()
