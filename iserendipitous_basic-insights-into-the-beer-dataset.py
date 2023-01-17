# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

df_1 = pd.read_csv('../input/beers.csv')

df_2 = pd.read_csv('../input/breweries.csv')

print(df_1.shape)

print(df_2.shape)

print(df_1.dtypes)

print(df_2.dtypes)
df_2['brewery_id'] = df_2.iloc[:,0]

df = pd.merge(df_1, df_2, on='brewery_id')

df.rename(columns={'name_x':'beer_name', 'name_y':'brewery_name'}, inplace=True)

df.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1, inplace=True)

print(df.dtypes)

print(df.shape)

print(df.head(10))
# checking the NaN values

print(df[pd.isnull(df).any(axis=1)].shape)

# there are lots of NaN values in this dataframe, but we will not fill out the NaN values instead we'll try to work with the given data
df_beer_count = df.groupby('brewery_name')[['beer_name']].count()

df_beer_count.columns = ['beer_name_count']

df_bc = df_beer_count.ix[df_beer_count.beer_name_count>10,:]

df_bc.plot.barh(title='brewing companies selling beer with 15 diffent beer names  in market', legend=True, figsize=(6,8))

plt.show()
df_state = df.groupby('state')[['brewery_name']].count()

df_state.columns = ['brewery_count']

df_maxbrewperstate = df_state.ix[df_state.brewery_count>50,:]

df_maxbrewperstate.plot.pie(y='brewery_count', autopct='%2f', title='states with % of breweries more than 50 ',legend=False ,figsize=(6,6)) 

plt.show()
df_by_abv = df.groupby('abv', as_index=False)[['brewery_name']].count()

df_by_abv.columns =['abv', 'count']

print(df_by_abv[pd.isnull(df_by_abv).any(axis=1)])

df_by_abv['categories'] = pd.cut(df_by_abv.abv,3, labels=['low_abv(0.00083-0.0433)', 'medium_abv(0.0433-0.0857)','high_abv(0.0857-0.128)'])

df_category_count = df_by_abv.groupby('categories')[['count']].count()

print(df_category_count.sum())

df_category_count.plot.barh(title='breweries count producing beer by abv content category', color=['green','blue','red'])

plt.show()
# breakdown of our dataframe to check sizes and to check NaN values in all three categories.

df['by_category'] = pd.cut(df.abv,3, labels=['low_abv', 'medium_abv', 'high_abv'])

print(df.shape, 'complete df size')

df_m = df[df['by_category']=='medium_abv']

print(df_m.shape, 'df size with medium_abv content')

print(df_m[pd.isnull(df_m).any(axis=1)].shape, 'medium_abv df shape with NaN values')

df_l = df[df['by_category']=='low_abv']

print(df_l.shape, 'df size with low_abv')

print(df_l[pd.isnull(df_l).any(axis=1)].shape, 'low_abv df shape with NaN values')

df_h = df[df['by_category']=='high_abv']

print(df_h.shape, 'df size with high_abv')

print(df_h[pd.isnull(df_h).any(axis=1)].shape, 'high_abv df shape with NaN values')

print(df[pd.isnull(df).any(axis=1)].shape, 'complete df shape with NaN values ')
df_state = pd.crosstab([df.state],df.by_category)

print(df_state.shape)

df_state.plot.barh(stacked=True, figsize=(8,10), title='medium abv content beer is most likeable in all the states')

plt.xlabel='total number of beer'

plt.show()
print(df[['abv']].corrwith(df['ibu']))

fig = plt.figure()

plt.scatter(df_l['abv'], df_l['ibu'], c='blue', marker='o', label='low_abv')

plt.scatter(df_m['abv'], df_m['ibu'], c='green', marker='x', label='medium_abv')

plt.scatter(df_h['abv'], df_h['ibu'], c='red', marker='^', label='high_abv')

plt.xlabel ='abv_content'

plt.ylabel = ('ibu level')

plt.legend()

plt.show()