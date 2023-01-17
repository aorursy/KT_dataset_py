import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib

import matplotlib.pyplot as plt

from scipy.stats import kendalltau

import warnings

matplotlib.style.use('ggplot')

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore")

%matplotlib inline

import seaborn as sns
train = pd.read_csv('../input/train_2016_v2.csv', parse_dates=["transactiondate"])

properties = pd.read_csv('../input/properties_2016.csv')
#merge data on key 

train_df_merged = pd.merge(train, properties, on='parcelid', how='left')

train_df_merged.head()
plt.figure(figsize=(12,8))

sns.distplot(train.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
train['transaction_month'] = train['transactiondate'].dt.month



cnt_srs = train['transaction_month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
train['transaction_day'] = train['transactiondate'].dt.day



cnt_srs = train['transaction_day'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color = "#9b59b6")

plt.xticks(rotation='vertical')

plt.xlabel('Day of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
geocolumns = [  'latitude', 'longitude'

                            ,'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc'

                            ,'regionidcity','regionidcounty', 'regionidneighborhood', 'regionidzip'

                            ,'censustractandblock', 'rawcensustractandblock']
import gc



geoproperties = properties[geocolumns]
plt.figure(figsize=(12,12))

sns.jointplot(x=properties.latitude.values, y=properties.longitude.values, size=10)

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

plt.show()
nan_df = train_df_merged.isnull().sum(axis=0).reset_index()

nan_df.columns = ['column_name', 'missing_count']

nan_df['missing_ratio'] = nan_df['missing_count'] / train_df_merged.shape[0]

nan_df.ix[nan_df['missing_ratio']>0.999]
col = "finishedsquarefeet12"

ulimit = np.percentile(train_df_merged[col].values, 99.5)

llimit = np.percentile(train_df_merged[col].values, 0.5)

train_df_merged[col].ix[train_df_merged[col]>ulimit] = ulimit

train_df_merged[col].ix[train_df_merged[col]<llimit] = llimit



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df_merged.finishedsquarefeet12.values, y=train_df_merged.logerror.values, size=10)

plt.ylabel('Log Error', fontsize=12)

plt.xlabel('Finished Square Feet 12', fontsize=12)

plt.title("Finished square feet 12 Vs Log error", fontsize=15)

plt.show()
print('Correlation with Log Error')



print(train_df_merged.corr(method='pearson').drop(['logerror']).sort_values('logerror', ascending=False)['logerror'].head(14))

print('\n')
#Assigning the new DF !



corr_series = train_df_merged.corr(method='pearson').drop(['logerror']).sort_values('logerror', ascending=False)['logerror'].head(14)
corr_df = pd.DataFrame(corr_df)

corr_df = corr_df.reset_index()

corr_df.columns = ['column_name', 'correlation']