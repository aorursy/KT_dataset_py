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
df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
diamonds = pd.DataFrame(df)
df.columns = ('index','carat','cut','color','clarity','depth','table','price',

              'x','y','z')
df
s = pd.Series(df['carat'])
s.sort_values(ascending=True, inplace=False, kind='quicksort', na_position='last')
print((df.x*diamonds.y*diamonds.z).head())
print(pd.concat([diamonds, diamonds.color], axis=1).head())
print(diamonds.loc[1,:])
print(diamonds.loc[[0,5,7],:])
print(diamonds.loc[2:6,:])
print(diamonds.loc[2:6,['color','price']])
print(diamonds.loc[0:2,'color':'price'])
print(diamonds.loc[diamonds.cut=='Premium','color'])
print(diamonds.iloc[[0,1],[0,3]])
print(diamonds.iloc[0:4,1:4])
print(diamonds.loc[1:3,:])
print(diamonds.iloc[2:5,1:2])
print(diamonds.info())
print(diamonds.info(memory_usage='deep'))
print(diamonds.memory_usage(deep=True))
print(diamonds.sample(n=5))
result = diamonds.sample(frac=0.75, random_state=99)

print(result)

print(diamonds.loc[~diamonds.index.isin(result.index), :])
print(diamonds.color.duplicated().sum())
print(diamonds.duplicated().sum())