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
from zipfile import ZipFile 

!unzip ../input/expedia-personalized-sort/data

df_train = pd.read_csv('train.csv',delimiter=',')
df_train.head()
df_train.shape
df_train.columns
df_train.isnull().sum()
df_train.info
df_train.describe
df_train['visitor_hist_adr_usd'].unique()
df_train['visitor_hist_starrating'].unique()
drop_list = ['srch_id', 'date_time', 'site_id', 'visitor_location_country_id',

       'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id',

       'prop_id','random_bool', 'comp1_rate', 'comp1_inv',

       'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',

       'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',

       'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',

       'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',

       'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',

       'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',

       'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',

       'comp8_rate_percent_diff', 'click_bool', 'gross_bookings_usd',

       'booking_bool']
df_train = df_train.drop(drop_list,axis=1)
df_train
df_train.shape
df_train.info()
df_train.describe()
cols = df_train.loc[: , "prop_location_score1":"prop_location_score2"]

df_train['prop_location_score'] = cols.mean(axis=1)

df_train


df_train.head()
df_train['promotion_flag'].unique()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
df_train.isnull().sum()
drop_list_again = ['srch_query_affinity_score','orig_destination_distance']

df_analyze = df_train.drop(drop_list_again,axis=1)
df_analyze
df_analyze.isnull().sum()
df_analyze['prop_review_score'].fillna((df_analyze['prop_review_score'].mean()),inplace=True)
df_analyze.isnull().sum()
df_analyze.head()
drop_list_new = ['prop_location_score1','prop_location_score2']
df_analyze = df_analyze.drop((df_analyze[drop_list_new]),axis=1)
df_analyze
df_analyze.describe()
kmeans.fit(df_analyze)
kmeans.cluster_centers_
print(kmeans.labels_)

print(len(kmeans.labels_))
print(type(kmeans.labels_))
unique,counts = np.unique(kmeans.labels_,return_counts=True)

print(dict(zip(unique,counts)))
import seaborn as sns

df_analyze['cluster'] = kmeans.labels_

sns.set_style('whitegrid')

sns.lmplot('prop_review_score','price_usd',data = df_analyze,hue = 'cluster',palette='coolwarm',size=6,aspect=1,fit_reg=False)
kmean_4 = KMeans(n_clusters=4)

kmean_4.fit(df_analyze.drop('position',axis=1))
print(kmean_4.cluster_centers_)

unique,counts = np.unique(kmean_4.labels_,return_counts=True)

kmean_4.cluster_centers_

print(dict(zip(unique,counts)))
import seaborn as sns

df_analyze['cluster'] = kmean_4.labels_

sns.set_style('whitegrid')

sns.lmplot('prop_review_score','price_usd',data = df_analyze,hue = 'cluster',palette='coolwarm',size=6,aspect=1,fit_reg=False)
kmean_8 = KMeans(n_clusters=16)

kmean_8.fit(df_analyze)

print(kmean_8.cluster_centers_)

unique,counts = np.unique(kmean_8.labels_,return_counts=True)

kmean_8.cluster_centers_

print(dict(zip(unique,counts)))
