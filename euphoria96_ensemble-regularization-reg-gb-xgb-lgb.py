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
# import 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_regression

import statsmodels.api as sm

from statsmodels.graphics import utils

from geopy.geocoders import Nominatim

from geopy.exc import GeocoderTimedOut

from itertools import combinations

from sklearn.model_selection import train_test_split

import scipy as sp

import datetime

import sys



import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
# data load

df_train = pd.read_csv('../input/train.csv')
df_train.shape
# data preprocessing

train_ID = df_train['id']



Xvar = list(df_train.columns[3:])

Xvar.insert(0,'date')

Xvar.insert(0,'price')



df_train = pd.DataFrame(df_train, columns=Xvar) # 그..그냥 칼럼 순서 바꾸기
#df_train = df_train.loc[:,list(df_train.columns[1:])]

df_train['date'] = df_train['date'].apply(lambda x:x[:6]).astype(int)

df_train['date'] = df_train['date'].apply(lambda x: x - 201500 + 8 if x > 201500 else x - 201404)



# log transformation for target variable

df_train['price'] = np.log1p(df_train['price'])
########### Check outliars



## Cook's Distance

# if D > 4/(N-K-1) then outliar

var = Xvar[1:]

data = pd.concat([df_train['price'], df_train[var]], axis=1)

model = sm.OLS(data['price'],data[var])

result = model.fit()

sm.graphics.plot_leverage_resid2(result)

plt.show()



# 

a = pd.merge(df_train.iloc[[1231,8756,8912]],

             pd.DataFrame(df_train.mean(axis=0)).transpose(),

             how='outer')

# 1231: sqft_lot, view, sqft_lot15 => delete

# 8756: bed/bathroom, sqft_living, sqft_lot, sqft_above, ... -> Including!

# 8912: bedrooms, bathrooms, sqft_living, sqft_lot, floors, view, grade, ... => delete





var = 'sqft_lot'

data = pd.concat([df_train['price'], df_train[var]], axis=1)

#

f, ax = plt.subplots(figsize=(6, 4.5))

fig = sns.regplot(x=var, y="price", data=data)



## outliars of locations

fig, axes = plt.subplots(figsize=(5,4))

sns.regplot(data=df_train, x='long', y='lat',

           fit_reg=False,

           scatter_kws={'s': 15},

           ax=axes)

axes.grid(False)

axes.set_title('Location of train data', fontsize=15)

plt.show()

## find address

from geopy.geocoders import Nominatim

geolocator = Nominatim()

# str(lat[0])+','+str(long[0]) # '47.5112,-122.257'

a = df_train.head(5)

is_seattle = pd.Series([geolocator.reverse(str(a['lat'][i])+','+str(a['long'][i])).

              address.find('Seattle') for i in range(a.shape[0])]).apply(lambda x: 0 if x == -1 else 1)