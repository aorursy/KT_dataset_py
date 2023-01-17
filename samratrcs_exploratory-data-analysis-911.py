# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
accdf = pd.read_csv('../input/911.csv')

# Any results you write to the current directory are saved as output.
#################### Hour wise emergency situation
accdf['hour'] = accdf['timeStamp'].str.split(" ").str[1].str.split(":").str[0]
accdf['hour']=accdf['hour'].astype(int)

accdf.loc[accdf.hour < 6, 'hourframe'] = '00 - 06'
accdf.loc[(accdf.hour >= 6) & (accdf.hour < 12), 'hourframe'] = '06 - 12'
accdf.loc[(accdf.hour >= 12) & (accdf.hour < 18), 'hourframe'] = '12 - 18'
accdf.loc[(accdf.hour >= 18) & (accdf.hour < 24), 'hourframe'] = '18 - 24'

accgbdf = pd.DataFrame(accdf.groupby('hourframe').size(), columns = ['count'])
accgbdf.reset_index(inplace = True)
sns.barplot(x = 'hourframe', y = 'count', data = accgbdf)
############################ Top 10 cities with most vehicle accidents
vehacdf = accdf.loc[accdf['title'].str.contains('Traffic: VEHICLE ACCIDENT')]
vehacdf = pd.DataFrame(vehacdf.groupby('twp').size(), columns = ['Count'])
vehacdf.reset_index(inplace = True)
vehacdf.sort_values('Count', inplace = True, ascending = False)
axvehac = sns.barplot(x = 'twp', y = 'Count', data = vehacdf.head(10))
axvehac.set_xticklabels(axvehac.get_xticklabels(), rotation = 90)
###################twp wise vehicle accident distn
accdf['hour'] = accdf['timeStamp'].str.split(" ").str[1].str.split(":").str[0]
accdf['hour']=accdf['hour'].astype(int)
twp = 'LOWER MERION'
twpaccdf = accdf.loc[accdf['twp'] == twp]
sns.distplot(a = twpaccdf['hour'], bins=24)
