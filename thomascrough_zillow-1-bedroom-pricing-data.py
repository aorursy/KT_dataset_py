# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ggplot import *

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/median_price.csv')

df.head()

## Noticed most regions lack consistent data up until August of 2015, retrieve this column location



print(df.columns.get_loc('2015-08'))

## Decided to look at Sept. 2015 to Sept. 2016

prices = df.columns[74:86]

df['yr_average'] = np.nanmean(df[prices], axis=1)

df['net_price_change_total'] = df['2016-09']-df['2010-01']

df['net_price_change_2yr'] = df['2016-09']-df['2014-09']

df['net_price_change_1yr'] = df['2016-09']-df['2015-09']

df['net_%_1yr'] = ((df['2016-09']-df['2015-09']) / df['2015-09'])*100

print(df[['City', 'yr_average','net_%_1yr','net_price_change_1yr']].nlargest(20, 'net_price_change_1yr'))

print("")

print("Data available for ", df['net_price_change_1yr'].count()," regions.")

avg_net_change_1yr = np.mean(df['net_price_change_1yr'])

avg_percent_1yr = np.mean(df['net_%_1yr'])

print("Average net change between 09/15 and 09/16: ", avg_net_change_1yr)

print("Average percent change between 09/15 and 09/16: ", avg_percent_1yr)



## Plot showing which cities contain regions with the greatest net price change



top_df = df[['City', 'yr_average','net_%_1yr','net_price_change_1yr']].nlargest(20, 'net_price_change_1yr')



r = sns.countplot(y='City', data=top_df, linewidth=3, palette='Blues')

print(r)
## Plot showing the counts of which cities had regions with the greatest percent 

## net change in property values







r = sns.countplot(y='City', data=top_df, linewidth=3, palette='BuGn_r')

print(r)
city_df = df.groupby(['City']).mean()

city_df[['yr_average','net_%_1yr','net_price_change_1yr']].nlargest(20, 'net_%_1yr')

city_df[['yr_average','net_%_1yr','net_price_change_1yr']].nsmallest(10, 'net_%_1yr')
p = sns.jointplot(x='yr_average', y='net_%_1yr', data=city_df, color="k")

    

print(p)
s = sns.stripplot(y='City', x='yr_average', data=df)