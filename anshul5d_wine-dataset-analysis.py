# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
wine= pd.read_csv('../input/winemag-data-130k-v2.csv')
wine =wine.drop(['Unnamed: 0'],axis=1)
wine.head()
wine.dtypes
wine.describe()
plt.figure(figsize=(16,8))
wine['country'].value_counts().plot.bar(x='index',y='country')
#wine_country_counts
wine.columns
plt.figure(figsize=(16,8))
wine['variety'].value_counts().head(43).plot.bar()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
wine_max = wine.groupby(['country'],as_index=True)['points'].max()
wine_max.head(50).plot.bar(ax=ax1)
wine_min = wine.groupby(['country'],as_index=True)['points'].min()
wine_min.head(50).plot.bar(ax=ax2)

plt.figure(figsize=(16,8))
wine_max_min=pd.DataFrame(wine_max)
wine_max_min['min'] = wine_min
col= {'points':'MAX_POINTS','min':'MIN_POINTS'}
wine_max_min=wine_max_min.rename(columns=col)
wine_max_min.plot.bar()
#wine_30_country = wine['country'].value_counts().head(30)
#wine.pivot(index='')
#wine_30_variety = wine['variety'].value_counts().head(30)
#wine.pivot(index='country',columns='variety',values='points')
#wine
sns.distplot(wine['points'])
tc_heat=wine.corr()
sns.heatmap(tc_heat)


