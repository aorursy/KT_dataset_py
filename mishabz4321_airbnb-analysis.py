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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
df= pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.info()
df.shape
df.describe().transpose()
df.isnull().sum()
plt.figure(figsize=(8,8))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
df['name'].value_counts()
df['host_name'].value_counts()
df['host_id'].value_counts()
df['last_review'].value_counts().head(10)
last_review_nan= df[df.last_review.isnull()]
last_review_nan.head()
last_review_nan.info()
last_review_nan[last_review_nan['number_of_reviews']==0].info()
df1= df.copy()
df1.drop(['name', 'host_name','last_review'], axis=1, inplace=True)
df1.isnull().sum()
df1.fillna({'reviews_per_month':0},inplace=True)
sns.heatmap(df1.isnull(),cmap='viridis', yticklabels=False, cbar=False)
df1.isnull().sum()
df1.head()
df1.describe().transpose()
sns.set_style('whitegrid')
plt.figure(figsize=(16,6))
sns.distplot(df1['price'], bins=30)
df2= df1.copy()
df1[df1['price'] ==0].head()
df2= df1[df1['price']!=0]
print(df1.shape)
print(df2.shape)
plt.figure(figsize=(10,8))
sns.boxplot(x='neighbourhood_group', y='price', data=df2)
brooklyn= df2[df2['neighbourhood_group']=='Brooklyn']
brooklyn_price= brooklyn[['price']]
manhattan= df2[df2['neighbourhood_group']=='Manhattan']
manhattan_price= manhattan[['price']]
queens= df2[df2['neighbourhood_group']=='Queens']
queens_price= queens[['price']]
staten_island= df2[df2['neighbourhood_group']=='Staten Island']
staten_island_price= staten_island[['price']]
bronx= df2[df2['neighbourhood_group']=='Bronx']
bronx_price= bronx[['price']]
brooklyn.head()
print('brookyln:', brooklyn.groupby('room_type')['price'].mean())
print('manhattan:', manhattan.groupby('room_type')['price'].mean())
print('queens:', queens.groupby('room_type')['price'].mean())
print('staten_island:', staten_island.groupby('room_type')['price'].mean())
print('bronx:', bronx.groupby('room_type')['price'].mean())
price_neigh_df= [brooklyn_price, manhattan_price, staten_island_price, queens_price, bronx_price]
neigh_list= ['brooklyn','manhattan', 'staten_island','queens','bronx']
price_neigh_stats=[]
for x in price_neigh_df:
    i =x.describe(percentiles=[.25,.50,.75])
    i= i[3:]
    i.reset_index(inplace=True)
    i.rename(columns={'index':'stats'}, inplace=True)
    price_neigh_stats.append(i)
price_neigh_stats[0].rename(columns={'price':neigh_list[0]}, inplace=True)
price_neigh_stats[1].rename(columns={'price':neigh_list[1]}, inplace=True)
price_neigh_stats[2].rename(columns={'price':neigh_list[2]}, inplace=True)
price_neigh_stats[3].rename(columns={'price':neigh_list[3]}, inplace=True)
price_neigh_stats[4].rename(columns={'price':neigh_list[4]}, inplace=True)
stats_df= price_neigh_stats
stats_df= [df.set_index('stats') for df in stats_df]
stats_df= stats_df[0].join(stats_df[1:])
stats_df
fig, axes= plt.subplots(ncols=1, nrows=5, figsize=(13,10))
plt.tight_layout()
sns.distplot(brooklyn['price'],ax=axes[0],kde_kws={'label':'brooklyn'})
sns.distplot(manhattan['price'],ax=axes[1],kde_kws={'label':'manhattan'})
sns.distplot(queens['price'],ax=axes[2],kde_kws={'label':'queens'})
sns.distplot(staten_island['price'], ax=axes[3],kde_kws={'label':'staten island'})
sns.distplot(bronx['price'],ax=axes[4],kde_kws={'label':'bronx'})
axes[0].set_xlabel('')
axes[1].set_xlabel('')
axes[2].set_xlabel('')
axes[3].set_xlabel('')
fig, axes= plt.subplots(ncols=1, nrows=5, figsize=(13,10))
plt.tight_layout()
sns.boxplot(x='room_type', y='price', data=brooklyn, ax=axes[0])
sns.boxplot(x='room_type', y='price', data=manhattan, ax=axes[1])
sns.boxplot(x='room_type', y='price', data=queens, ax=axes[2])
sns.boxplot(x='room_type', y='price', data=staten_island, ax=axes[3])
sns.boxplot(x='room_type', y='price', data=bronx, ax=axes[4])
axes[0].set_xlabel('')
axes[1].set_xlabel('')
axes[2].set_xlabel('')
axes[3].set_xlabel('')
axes[0].legend(['brooklyn'])
axes[1].legend(['manhattan'])
axes[2].legend(['queens'])
axes[3].legend(['staten_island'])
axes[4].legend(['bronx'])
from scipy import stats
def remove_price_outliers(df):
    df_out= pd.DataFrame()
    for key, subdf in brooklyn.groupby('room_type'):
        subdf['zscore_price']= np.abs(stats.zscore(subdf.price))
        reduced_df= subdf[(subdf.zscore_price>-2)&(subdf.zscore_price<2)]
        df_out= pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
brooklyn_no_outlier= remove_price_outliers(brooklyn)
def remove_price_outliers(df):
    df_out= pd.DataFrame()
    for key, subdf in manhattan.groupby('room_type'):
        subdf['zscore_price']= np.abs(stats.zscore(subdf.price))
        reduced_df= subdf[(subdf.zscore_price>-2)&(subdf.zscore_price<2)]
        df_out= pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
manhattan_no_outlier= remove_price_outliers(manhattan)
def remove_price_outliers(df):
    df_out= pd.DataFrame()
    for key, subdf in queens.groupby('room_type'):
        subdf['zscore_price']= np.abs(stats.zscore(subdf.price))
        reduced_df= subdf[(subdf.zscore_price>-2)&(subdf.zscore_price<2)]
        df_out= pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
queens_no_outlier= remove_price_outliers(queens)
def remove_price_outliers(df):
    df_out= pd.DataFrame()
    for key, subdf in staten_island.groupby('room_type'):
        subdf['zscore_price']= np.abs(stats.zscore(subdf.price))
        reduced_df= subdf[(subdf.zscore_price>-2)&(subdf.zscore_price<2)]
        df_out= pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
staten_island_no_outlier= remove_price_outliers(staten_island)
def remove_price_outliers(df):
    df_out= pd.DataFrame()
    for key, subdf in bronx.groupby('room_type'):
        subdf['zscore_price']= np.abs(stats.zscore(subdf.price))
        reduced_df= subdf[(subdf.zscore_price>-2)&(subdf.zscore_price<2)]
        df_out= pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
bronx_no_outlier= remove_price_outliers(bronx)
df3= pd.concat([brooklyn_no_outlier,manhattan_no_outlier,queens_no_outlier,staten_island_no_outlier, bronx_no_outlier], axis=0)
df3.shape
df2.shape
sns.boxplot(x='room_type', y='price', data=brooklyn)
sns.boxplot(x='room_type', y='price', data=brooklyn_no_outlier)
plt.figure(figsize=(10,8))
sns.boxplot(x='neighbourhood_group', y='price', data=df3)
plt.figure(figsize=(14,8))
sns.distplot(df3['price'])
df3.head()
df3['neighbourhood']= df3['neighbourhood'].apply(lambda x: x.strip())
location_stats= df3['neighbourhood'].value_counts(ascending=False)
location_stats.head(20)
location_stats_lessthan_10= location_stats[location_stats<10]
location_stats_lessthan_10.head(20)
plt.figure(figsize=(10,8))
viz1=sns.violinplot(x='neighbourhood_group', y='price', data=df3)
viz1.set_title('density and distribution of price in the neighbourhood group')
df4= df3.copy()
df4['neighbourhood']= df4['neighbourhood'].apply(lambda x : 'other' if x in location_stats_lessthan_10 else x)
len(df3['neighbourhood'].unique())
len(df4['neighbourhood'].unique())
df4.groupby('neighbourhood_group')['number_of_reviews'].mean().sort_values(ascending=False)
df4['calculated_host_listings_count'].value_counts().head()
df4['calculated_host_listings_count'].max()
top_host= df4['host_id'].value_counts().head(10)
top_host
top_host_df= pd.DataFrame(top_host)
top_host_df.reset_index(inplace=True)
top_host_df.rename(columns={'index':'host_id', 'host_id':'p_count'}, inplace=True)
top_host_df
g= sns.barplot(x='host_id', y='p_count', data=top_host_df, palette='Blues_d')
g.set_title('hosts with most listings in NYC')
g.set_ylabel('count of listings')
g.set_xlabel('Host_Ids')
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.figure(figsize=(16,10))
sns.scatterplot(x='latitude', y='longitude', data=df3, hue='neighbourhood_group')
fig, axes= plt.subplots(ncols=5, nrows=1, figsize=(20,6))
plt.tight_layout()
sns.scatterplot(x='latitude', y='longitude', data=brooklyn_no_outlier, hue='price', size='price', ax=axes[0])
sns.scatterplot(x='latitude', y='longitude', data=manhattan_no_outlier, hue='price', size='price', ax=axes[1])
sns.scatterplot(x='latitude', y='longitude', data=queens_no_outlier, hue='price', size='price', ax=axes[2])
sns.scatterplot(x='latitude', y='longitude', data=staten_island_no_outlier, hue='price', size='price', ax=axes[3])
sns.scatterplot(x='latitude', y='longitude', data=bronx_no_outlier, hue='price', size='price', ax=axes[4])
df4['availability_365'].value_counts().head()
viz2= df4.plot(kind='scatter', x='latitude', y='longitude', label='availability_365', c='price',
               cmap=plt.get_cmap('jet'), colorbar= True, alpha=0.4, figsize=(10,8))
viz2.legend()
import urllib
plt.figure(figsize=(15,8))
i= urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')
nyc_img= plt.imread(i)
plt.imshow(nyc_img, zorder=0, extent=[df4['longitude'].min(), df4['longitude'].max(), df4['latitude'].min(),df4['latitude'].max()])
ax=plt.gca()
df4.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
             ax=ax,cmap=plt.get_cmap('jet'), colorbar= True, alpha=0.4, figsize=(10,8), zorder=5)
plt.legend()
plt.show()
top_reviewed_listings= df4.nlargest(10,'number_of_reviews')
top_reviewed_listings.head()
brooklyn_neighbourhood= pd.DataFrame(brooklyn_no_outlier['neighbourhood'].value_counts().head(10))
brooklyn_neighbourhood.reset_index(inplace=True)
brooklyn_neighbourhood.rename(columns={'index':'brooklyn', 'neighbourhood':'p_counts'}, inplace=True)
g1= sns.barplot(x='brooklyn', y='p_counts', data=brooklyn_neighbourhood, palette='Blues_d')
g1.set_title('top listings in brooklyn')
g1.set_xticklabels(g1.get_xticklabels(), rotation=90)
manhattan_neighbourhood= pd.DataFrame(manhattan_no_outlier['neighbourhood'].value_counts().head(10))
manhattan_neighbourhood.reset_index(inplace=True)
manhattan_neighbourhood.rename(columns={'index':'manhattan', 'neighbourhood':'p_counts'}, inplace=True)
g2= sns.barplot(x='manhattan', y='p_counts', data=manhattan_neighbourhood, palette='Blues_d')
g2.set_title('top listings in manhattan')
g2.set_xticklabels(g1.get_xticklabels(), rotation=90)
df4['neighbourhood'].value_counts().head(10)
top_neighbourhood= ['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick','Hell\'s Kitchen', 
                    'Upper West Side','East Village','Upper East Side','Crown Heights','Midtown']
df5= df4.copy()
df5= df4[df4['neighbourhood'].isin(top_neighbourhood)]
df5.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)
plt.figure(figsize=(20,6))
sns.distplot(df5[df5['neighbourhood']=='Williamsburg']['price'],kde_kws={'color':'b', 'label':'Williamsburg'})
sns.distplot(df5[df5['neighbourhood']=='Bedford-Stuyvesant']['price'],kde_kws={'color':'r', 'label':'Bedford-Stuyvesant'})
sns.distplot(df5[df5['neighbourhood']=='Harlem']['price'],kde_kws={'color':'g', 'label':'Harlem'})