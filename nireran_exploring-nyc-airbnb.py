# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

import seaborn as sns
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gzip
import os
print(os.listdir('../input/listings.csv'))

# Any results you write to the current directory are saved as output.
!cat '../input/listings.csv'
pd.options.display.max_columns=100
dflisting = pd.read_csv('../input/listings.csv/listings.csv' ,  parse_dates=['host_since','first_review','last_review'])
dflisting.head()



#choosing potenial columns 

cols = [col for col in dflisting.columns if col not in ['name','summary','description','house_rules','host_id','host_since','host_is_superhost','host_total_listings_count','neighbourhood_cleansed','neighbourhood_group_cleansed','latitude','longitude','property_type','room_type','bedrooms','bed_type','price','minimum_nights','availability_365','number_of_reviews','first_review','last_review','review_scores_rating','reviews_per_month','accommodates','calculated_host_listings_count']]
df=dflisting.drop(cols ,axis=1).reset_index()
# renaming long column names
df = df.rename(columns={ 
                            'host_is_superhost':'superhost', 
                            'neighbourhood_cleansed':'neighbourhood', 
                            'neighbourhood_group_cleansed': 'neighbourhood_group'})



df.info()
df.isnull().sum()
#cleaning price column
df['price']=df['price'].replace('[$,]','',regex=True).astype(float)

df['price'].head()
# removing neighbourhood records with less than 30 listings 
value_counts=df.neighbourhood.value_counts()
to_remove = value_counts[value_counts < 30].index
print(to_remove)  #neighbourhood that will be removed 
df = df[~df.neighbourhood.isin(to_remove)]
#Clean bed_type to binary real bed or other
def bed(value):
    if value not in ['Real Bed']:
        return 'Other'
    return value

df['bed_type'] = df['bed_type'].apply(bed)

df['real_bed']=df['bed_type'].map({'Other':0,'Real Bed':1})

df['real_bed'].unique()


# if superhost is missing (6 records) I assume that they are not + change to boolean 0,1
df['superhost']=df['superhost'].fillna('f').map({'f':0,'t':1})
df['superhost'].unique()
# I wont use thse columns this time.. 
df=df.drop(['bed_type','name','summary','description','house_rules'], axis=1)
#removing 9 records with host_since = null
df=df[~df.host_since.isna()]
df['bedrooms']=df['bedrooms'].fillna(1) 
df.info()                            
df.isnull().sum()
df.loc[(~df['reviews_per_month'].isna()) & (df['review_scores_rating'].isna())][['reviews_per_month','review_scores_rating','neighbourhood','host_id']].tail()
s=df.groupby('neighbourhood')['review_scores_rating'].transform(np.mean)

df.review_scores_rating.mean()
#replace null with neighbourhood mean 

#df.loc[((~df['reviews_per_month'].isna()) & (df['review_scores_rating'].isna())),['review_scores_rating']].fillna(s,inplace=True)
df.loc[((~df['reviews_per_month'].isna()) & (df['review_scores_rating'].isna())),['review_scores_rating']]=93.71965648347557

df.isna().sum()
plt.scatter(df['neighbourhood_group'],df['price'])
#df=df[(df['neighbourhood_group']=='Manhattan')|(df['neighbourhood_group']=='Brooklyn')]
df = df[np.abs(df.price-df.price.mean())<=(3*df.price.std())]
plt.scatter(df['neighbourhood_group'],df['price'])
df=df[(df['neighbourhood_group']=='Manhattan')|(df['neighbourhood_group']=='Brooklyn')]
# calculating the time till first review in mobths + converting to numeric
df['timeToFirstReview']= ((df['first_review']-df['host_since'])/np.timedelta64(1, 'M'))

df['timeToFirstReview']=df['timeToFirstReview'].fillna(-999)
df[df['timeToFirstReview']>0]['timeToFirstReview'].describe().T



df_forplot=df[['review_scores_rating','number_of_reviews','price','bedrooms','timeToFirstReview','accommodates']]
df_forplot=df_forplot[df_forplot['timeToFirstReview']>0]
df_forplot=df_forplot.dropna()
df_forplot.info()
df_forplot['timeToFirstReview_grade']=pd.qcut(df_forplot['timeToFirstReview'],7,labels=['A','B','C','D','E','F','G'])

#df_forplot=df_forplot.dropna()
df_forplot.head(9)
#df[df['timeToFirstReview']>=0]['timeToFirstReview'].cut(bins='6',labels=['A','B','C','D','E','F'])
#df_forplot['timeToFirstReview']=np.log10(df['timeToFirstReview'])




sns.pairplot(df_forplot, dropna=True, palette='PuBu' ,hue='timeToFirstReview_grade', hue_order=['A','B','C','D','E','F'])
# option B with matplotli
#plt.figure(figsize=(10,10))
#df_forplot=df_forplot[['review_scores_rating','number_of_reviews','price','bedrooms','timeToFirstReview']]
#ax=plt.gca()
#pd.plotting.scatter_matrix(df_forplot,ax=ax)
#plt.show()
df[df['price']>500].count()
df[df['price']>500].count().div(len(df))  # only 1.5% of of total listing are priced higer than $ 500   

fig, axes = plt.subplots(figsize=(20,5) ,nrows=1, ncols=2)

for i,n in enumerate(df.neighbourhood_group.unique()):
    axes[i].set_title(n)
    temp=df[df['neighbourhood_group']==n]
    df_avg=temp[['neighbourhood_group','price','neighbourhood']]
    df_avg[df_avg['neighbourhood_group']==n].groupby('neighbourhood').mean().sort_values('price').plot(ax=axes[i] , kind="bar")
   
    

ax=df.plot(kind="scatter", x="longitude", y="latitude", c='price' , cmap=plt.get_cmap("jet"), vmax=400 ,colorbar=True ,alpha=0.4, figsize=(10,7), s=3 )

plt.ylabel("", fontsize=14)
#plt.xlabel("", fontsize=14)
plt.tick_params(colors='w')

plt.show()

fig, ax = plt.subplots(figsize=(10,10))

long_max = df['longitude'].max() + .02
long_min = df['longitude'].min() -.02
mid_long = (df['longitude'].min() + df['longitude'].max())/2

lat_max = df['latitude'].max() + .02
lat_min = df['latitude'].min() - .02
mid_lat = (df['latitude'].min() + df['latitude'].max())/2

## map projection='cyl'
m = Basemap(ax=ax,lat_0=mid_lat,lon_0=mid_long,\
            llcrnrlat=lat_min,urcrnrlat=lat_max,\
            llcrnrlon=long_min,urcrnrlon=long_max,\
            rsphere=6371200.,resolution='h',area_thresh=10)
m.drawcoastlines()
m.drawstates()
m.drawcounties()
m.shadedrelief()


ax.scatter(df['longitude'], df['latitude'],
           c=df['price'],alpha=0.5, zorder=10 ,s=4,vmax=400,cmap=plt.get_cmap("jet"))


#plt.ylabel("", fontsize=14)
#plt.xlabel("", fontsize=14)
plt.tick_params(colors='w')

plt.show()

fig, axes = plt.subplots(figsize=(20,5) ,nrows=1, ncols=2)
#fig = plt.figure(figsize=(15,50))

df1=df[(df['neighbourhood_group'] == ('Manhattan'))].groupby(['neighbourhood','room_type']).size().reset_index(name='norm')
a=df1.groupby('neighbourhood')['norm'].transform('sum')
df1['norm']=df1['norm'].div(a)
df1=df1.pivot(index='neighbourhood' , columns='room_type',values='norm')
#df1.head()
axes[0].set_title('Manhattan')
df1.plot(ax=axes[0] , kind='bar', stacked =True)

df1=df[(df['neighbourhood_group'] == ('Brooklyn'))].groupby(['neighbourhood','room_type']).size().reset_index(name='norm')
a=df1.groupby('neighbourhood')['norm'].transform('sum')
df1['norm']=df1['norm'].div(a)
df1=df1.pivot(index='neighbourhood' , columns='room_type',values='norm')
#df1.head()
axes[1].set_title('Brooklyn')
df1.plot(ax=axes[1] , kind='bar', stacked =True)
d={'price': ['mean' , 'count']}
price_property=df[['price','property_type']]
price_property_mean=price_property.groupby('property_type').agg(d).round(2)
price_property_mean 
def PropertyType_transform(x,PropertyType_value_count):
        
    if (PropertyType_value_count[x] > 20):
        return x
    else:
        return 'Other'

PropertyType_value_count = df["property_type"].value_counts()
df['property_type'] = df['property_type'].apply(lambda x: PropertyType_transform(x,PropertyType_value_count))

# check the mean listing price for each room type after mapping 
price_property=df[['price','property_type']]
price_property_mean=price_property.groupby('property_type').agg(d).round(2)
price_property_mean.sort_values(by=('price','mean'),ascending=False) 
# Explorer the relationship between price and property type
#df =df.drop([df['price']>500])
plt.figure(figsize=(10,8))
sns.boxplot(x='property_type',y='price',data=df)
ax=plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(df.groupby(['property_type', 'bedrooms']).price.mean().unstack(), annot=True, fmt=".0f" , cmap="Reds")
df['superhost'].unique()

df1=df[['price','superhost','number_of_reviews','review_scores_rating']]
df1.groupby('superhost').mean().T.plot(kind='bar' ,rot=45)
# Remove rows without value in host since (6 records)

df.host_since.isna().sum()
df = df[~df['host_since'].isnull()]

plt.figure(figsize=(10,20))

monthly_summary = pd.DataFrame()

df_HostSince=df[['host_since','neighbourhood_group']]
#df_HostSince['ind']=1
df_HostSince=df_HostSince.set_index('host_since')

monthly_summary['sampled'] = df_HostSince.neighbourhood_group.resample('M').count()
#monthly_summary.head(50)
monthly_summary.plot()



df.host_since.isna().sum()
df = df[~df['host_since'].isnull()]

plt.figure(figsize=(10,20))

monthly_summary = pd.DataFrame()
fig = plt.figure(figsize = (8,8))
ax = fig.gca()

df_HostSince=df[['host_since','neighbourhood_group']]
#df_HostSince['ind']=1
df_HostSince=df_HostSince.set_index('host_since')
df_HostSince=df_HostSince.groupby('neighbourhood_group').resample('M').size()
df_HostSince=df_HostSince.rolling(window=5).mean()
df_HostSince.unstack().T.plot(ax=ax)
              
              
#df_HostSince[(df_HostSince['neighbourhood_group'] == ('Manhattan'))].head(30)
#df_HostSince[df_HostSince['neighbourhood_group']=='Brooklyn'].plot(x='host_since')
#monthly_summary['ind'] = df_HostSince.ind.resample('M').sum()
#monthly_summary.head(50)
#monthly_summary.plot()




df.host_since.isna().sum()
df = dflisting[~dflisting['host_since'].isnull()]

plt.figure(figsize=(10,20))

monthly_summary = pd.DataFrame()
fig = plt.figure(figsize = (8,8))
ax = fig.gca()

df_HostSince=df[['host_since','neighbourhood_group_cleansed']]
#df_HostSince['ind']=1
df_HostSince=df_HostSince.set_index('host_since')
df_HostSince=df_HostSince.groupby('neighbourhood_group_cleansed').resample('M').size()
df_HostSince=df_HostSince.rolling(window=5).mean()
df_HostSince.unstack().T.plot(ax=ax)
              
              





