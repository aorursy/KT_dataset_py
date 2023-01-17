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
import json

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

import seaborn as sns # a good library to graphic plots



from sklearn.cluster import KMeans

from sklearn import preprocessing



pd.set_option('display.max_columns', 500)
def load_df(csv_path='/kaggle/input/ga-customer-revenue-prediction/train_v2.csv', nrows=None):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    

    df = pd.read_csv(csv_path, 

                     converters={column: json.loads for column in JSON_COLUMNS}, 

                     dtype={'fullVisitorId': 'str'}, # Important!!

                     nrows=nrows)

    

    for column in JSON_COLUMNS:

        column_as_df = json_normalize(df[column])

        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]

        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df
%%time

train_df = load_df(nrows=10000)

#test_df = load_df("/kaggle/input/ga-customer-revenue-prediction/test_v2.csv")
train_df.head()
print("Number of unique visitors in train set : ",train_df.fullVisitorId.nunique(), " out of rows : ",train_df.shape[0])
ga=train_df.drop_duplicates(subset='fullVisitorId', keep='first', inplace=False)

ga1=ga.iloc[0:4295]

ga1.shape
ga2=ga1[['channelGrouping','device.deviceCategory','totals.visits','totals.hits','totals.pageviews']]

print(ga2.isnull().sum())

ga2.to_csv (r'ga.csv', index = None, header=True)
const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ]

print(len(const_cols))

print(const_cols)
train_df[const_cols].head()
cols_to_drop = const_cols + ['hits', 'customDimensions']



train_df1 = train_df.drop(cols_to_drop, axis=1)

print(train_df1.shape)

train_df1.head()
train_df1.dtypes
train_df1.isnull().sum()/len(train_df1)*100
train_df1[pd.notnull(train_df1["trafficSource.adwordsClickInfo.isVideoAd"])].head()
cols_to_drop1=['totals.transactions', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.page', 

               'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.gclId', 

               'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.isVideoAd', 

               'totals.bounces', 'trafficSource.keyword', 'trafficSource.referralPath', 'trafficSource.isTrueDirect']

train_df2 = train_df1.drop(cols_to_drop1, axis=1)
train_df2.isnull().sum()/len(train_df2)*100
# Impute 0 for missing target values

train_df2["totals.transactionRevenue"].fillna(0, inplace=True)

train_df2["totals.totalTransactionRevenue"].fillna(0, inplace=True)
train_df2.loc[train_df1["totals.transactionRevenue"] != 0].head()
train_df2['totals.newVisits'].unique()
train_df2.dtypes
train_df2['totals.newVisits'].fillna(0, inplace=True)

train_df2['totals.sessionQualityDim'].fillna(0, inplace=True)

train_df2['totals.timeOnSite'].fillna(0, inplace=True)

train_df2['totals.pageviews'].fillna(0, inplace=True)



train_df2.isnull().sum()/len(train_df2)*100
train_df2['fullVisitorId'].value_counts()
train_df2.loc[train_df2['fullVisitorId']=='1572225825161580042']
train_df2.drop_duplicates(subset='fullVisitorId', keep='first', inplace=True)

print(train_df2.shape)
cols_to_drop2=['date', 'visitId', 'visitNumber', 

               'visitStartTime', 'totals.transactionRevenue', 

               'totals.totalTransactionRevenue']

train_df3 = train_df2.drop(cols_to_drop2, axis=1)

print(train_df3.shape)

train_df3.head()
train_df3.set_index('fullVisitorId', inplace=True)

train_df3.head()
train_df3.dtypes
train_df4=train_df3.copy()



# label encode the categorical variables and convert the numerical variables to float

cat_cols = ["channelGrouping", "device.browser", 

            "device.deviceCategory", "device.operatingSystem", 

            "geoNetwork.city", "geoNetwork.continent", 

            "geoNetwork.country", "geoNetwork.metro",

            "geoNetwork.networkDomain", "geoNetwork.region", 

            "geoNetwork.subContinent", "trafficSource.campaign", 

            "trafficSource.source", 

            "trafficSource.medium"]



for col in cat_cols:

    print(col)

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(train_df3[col].values.astype('str')))

    train_df4[col] = lbl.transform(list(train_df3[col].values.astype('str')))

    #test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))





num_cols = ["totals.hits", "totals.pageviews", "totals.timeOnSite", 'totals.sessionQualityDim',  'totals.newVisits']    

for col in num_cols:

    train_df4[col] = train_df3[col].astype(float)

    #test_df[col] = test_df[col].astype(float)
train_df4.dtypes
train_df4.head()
kmeans =  KMeans(n_clusters = 5)

kmeans.fit(train_df4)
unique, counts = np.unique(kmeans.labels_, return_counts=True)

dict(zip(unique, counts))
cluster = pd.DataFrame(kmeans.labels_, columns=['cluster'], index=train_df4.index)

cluster.head()
df_combine = pd.concat([train_df3, cluster], axis = 1)

df_combine.head()
num_cols = ["totals.hits", "totals.pageviews", "totals.timeOnSite", 'totals.sessionQualityDim',  'totals.newVisits']    

for col in num_cols:

    df_combine[col] = df_combine[col].astype(float)

    

df_combine.dtypes
g = sns.catplot(x="channelGrouping", col="cluster", data=df_combine, kind="count", col_wrap=3, height=10)

g.set_xticklabels(fontsize=25, rotation=70)

g.set_yticklabels(fontsize=25)

g.set_xlabels(fontsize=25)

g.set_ylabels(fontsize=25)

g.set_titles(size=25)
g = sns.catplot(x="device.deviceCategory", col="cluster", data=df_combine, kind="count", col_wrap=3, height=10)

g.set_xticklabels(fontsize=25, rotation=70)

g.set_yticklabels(fontsize=25)

g.set_xlabels(fontsize=25)

g.set_ylabels(fontsize=25)

g.set_titles(size=25)
g = sns.catplot(x="geoNetwork.continent", col="cluster", data=df_combine, kind="count", col_wrap=3, height=10)

g.set_xticklabels(fontsize=25, rotation=70)

g.set_yticklabels(fontsize=25)

g.set_xlabels(fontsize=25)

g.set_ylabels(fontsize=25)

g.set_titles(size=25)
plt.figure(figsize=(14,7))

sns.boxplot(x="cluster", y="totals.hits", data=df_combine)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("cluster",fontsize=16)

plt.ylabel("totals.hits",fontsize=16)

plt.ylim(0,80)

plt.show()
plt.figure(figsize=(14,7))

sns.boxplot(x="cluster", y="totals.pageviews", data=df_combine)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("cluster",fontsize=16)

plt.ylabel("totals.pageviews",fontsize=16)

plt.ylim(0,70)

plt.show()
plt.figure(figsize=(14,7))

sns.boxplot(x="cluster", y="totals.timeOnSite", data=df_combine)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("cluster",fontsize=16)

plt.ylabel("totals.timeOnSite",fontsize=16)

#plt.ylim(0,70)

plt.show()