# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_terror = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
df_terror.head()
df_terror.info()
terror=df_terror[['iyear','imonth','iday','country_txt','region_txt','city','latitude','longitude',
                  'attacktype1_txt','nkill','nwound','target1','summary','gname','targtype1_txt', 
                  'targsubtype1_txt', 'weaptype1_txt','motive','specificity']]
print(terror.columns)
terror.columns = ['year', 'month', 'day', 'country', 'region', 'city',
       'latitude', 'longitude', 'attack_type', 'killed', 'wounded',
       'target', 'summary', 'group', 'target_type', 'target_sub_type','weapon_type',
       'motive', 'specifity']
terror.info()
f,ax = plt.subplots(figsize = (15, 15))
corr_data = terror.corr()
sns.heatmap(corr_data, annot=True, linewidths=0.5, fmt='.1f', ax=ax)
plt.show()
terror.head()
terror.plot(x='year', y='killed', kind='line',color = 'r',label = 'Killed',linewidth=1,
            alpha = 0.5,grid = True,linestyle = ':')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.title('Line Plot')            # title = title of plot
plt.show()
terror.Year.plot(kind = 'hist',bins = 10,figsize = (12,12))
plt.show()
series = terror['Country']        # data['Defense'] = series
print(type(series))
data_frame = terror[['Region']]  # data[['Defense']] = data frame
print(type(data_frame))
terror.region.unique()
terror[terror['Killed']>100].head()
terror[np.logical_and(terror['region']=='Middle East & North Africa', terror['killed']>100)].head()
terror.shape
terror.describe()
terror.month.value_counts()
terror.boxplot(column="Killed")
plt.show()
terror.dtypes
terror.attack_type.value_counts(dropna=False)
terror.loc[0,"Killed"]
terror.killed[0]
terror[['killed', 'wounded']]
terror.loc[0:10, "killed":]
terror.killed[terror.region=='Middle East & North Africa'].mean()
df_ndf_terror.nwound.head()
terror.index.name = "index_name" 
terror.head()
terror.tail()
terror.index = range(1,181692,1)
terror.head()
