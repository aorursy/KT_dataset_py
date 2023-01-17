# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/anime.csv')

data.info()
print(data['type'].value_counts(dropna=True))
data.describe()
data.boxplot(column = 'rating', by='type', figsize = (13,13))
#id_vars is what we don't want to wish to melt

#value_vars is what we want to wish to melt

data_new = data.head()

melted = pd.melt(frame=data_new, id_vars = 'name', value_vars =['genre', 'type'])

melted
melted.pivot(index = 'name', columns = 'variable', values = 'value')
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index=True)#axis=0 is meaning, vertical concat.

conc_data_row
data_conc_cols = pd.concat([data1,data2], axis = 1 , ignore_index=True)#axis = 1 is meaning, horizontal cocnat.

data_conc_cols
data.dtypes
data['type'] = data['type'].astype('category')

data['anime_id'] = data['anime_id'].astype('float')

data.dtypes
data.info()

#there are 12294 object in out dataframe.

#but as we can see, there are 12064 rating value at dataframe
data['rating'].value_counts(dropna=False)

#there are 230 NaN value.
dataNew = data

dataNew['rating'].dropna(inplace =True)
assert 1 == 1 # returns nothing.
assert 1 == 2 # returns error.
assert dataNew['rating'].dropna().all()#returns nothing because of we dropped all NaN values already.