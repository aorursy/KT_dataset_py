# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_selection import f_classif
#import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train_df.csv')
train_df.drop('Unnamed: 0',axis=1,inplace=True,errors='ignore')
train_df.head()
copy_train_df = train_df.copy()
const_cols = [c for c in copy_train_df.columns if copy_train_df[c].nunique(dropna=False)==1 ]
cols_to_drop = const_cols + ['sessionId'] + ['visitId']+["trafficSource.campaignCode"]

copy_train_df = copy_train_df.drop(cols_to_drop, axis=1)
#outliers
# fig, ax = plt.subplots()
# ax.scatter(x = copy_train_df['visitNumber'], y = copy_train_df['totals.transactionRevenue'])
# plt.ylabel('totals.transactionRevenue', fontsize=13)
# plt.xlabel('visitNumber', fontsize=13)
# plt.show()
# fig, ax = plt.subplots()
# ax.scatter(x = copy_train_df['totals.hits'], y = copy_train_df['totals.transactionRevenue'])
# plt.ylabel('totals.transactionRevenue', fontsize=13)
# plt.xlabel('totals.hits', fontsize=13)
# plt.show()
# fig, ax = plt.subplots()
# ax.scatter(x = copy_train_df['totals.pageviews'], y = copy_train_df['totals.transactionRevenue'])
# plt.ylabel('totals.transactionRevenue', fontsize=13)
# plt.xlabel('totals.pageviews', fontsize=13)
# plt.show()
# fig, ax = plt.subplots()
# ax.scatter(x = copy_train_df['date'], y = copy_train_df['totals.transactionRevenue'])
# plt.ylabel('totals.transactionRevenue', fontsize=13)
# plt.xlabel('date', fontsize=13)
# plt.show()
#missing value of numerical cols
copy_train_df["totals.transactionRevenue"].fillna(0, inplace=True)
copy_train_df['totals.pageviews'].fillna(0, inplace=True)
copy_train_df['trafficSource.adwordsClickInfo.page'].fillna(0, inplace=True)

# change date to get only month
def changeDateToMonth(ts):
    return str(ts)[4:6]
copy_train_df['date'] = copy_train_df['date'].apply(changeDateToMonth)

# changing visitStartTime from POSIX timestamp to hour
def changePOSIXtoHour(ts):
    return datetime.utcfromtimestamp(ts).strftime('%H')
copy_train_df['visitStartTime'] = copy_train_df['visitStartTime'].apply(changePOSIXtoHour)

# replace 'totals.bounces' : 1->0, nan->1
def switch(ts):
    if ts == 1:
        return 0
    else:
        return 1
copy_train_df['totals.bounces'] = copy_train_df['totals.bounces'].apply(switch)

def isNewVisit(val):
    if(val != 1):
        return 0
    else:
        return 1
copy_train_df['totals.newVisits'] = copy_train_df['totals.newVisits'].apply(isNewVisit)

def isRevenue(val):
    if (val >0):
        return 1
    else:
        return 0
copy_train_df['isRevenue'] = copy_train_df['totals.transactionRevenue'].apply(isRevenue) 
copy_train_df['isRevenue'] = copy_train_df['isRevenue'].astype(str)
#missing value in categorical cols
cols = ['trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adContent','trafficSource.adwordsClickInfo.gclId',
       'trafficSource.adwordsClickInfo.isVideoAd','trafficSource.adwordsClickInfo.slot',
       'trafficSource.isTrueDirect','trafficSource.keyword','trafficSource.referralPath']
for col in cols:
    copy_train_df[col].fillna('missing', inplace=True)
#target feature
from scipy import stats
from scipy.stats import norm, skew #for some statistics

# sns.distplot(copy_train_df['totals.transactionRevenue'] ,fit=norm);

# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(copy_train_df['totals.transactionRevenue'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('totals.transactionRevenue distribution')

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(copy_train_df['totals.transactionRevenue'], plot=plt)
# plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
copy_train_df["totals.transactionRevenue"] = np.log1p(copy_train_df["totals.transactionRevenue"])

# #Check the new distribution 
# sns.distplot(copy_train_df['totals.transactionRevenue'] , fit=norm);

# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(copy_train_df['totals.transactionRevenue'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('totals.transactionRevenue distribution')

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(copy_train_df['totals.transactionRevenue'], plot=plt)
# plt.show()
#Transforming some numerical variables that are really categorical
# copy_train_df['totals.bounces'] = copy_train_df['totals.bounces'].apply(str)
# copy_train_df['totals.newVisits'] = copy_train_df['totals.newVisits'].apply(str)

num = ['visitNumber','totals.hits','totals.pageviews',
       'trafficSource.adwordsClickInfo.page']
one_zero = ['device.isMobile','trafficSource.adwordsClickInfo.isVideoAd','trafficSource.isTrueDirect',
           'isRevenue','totals.bounces','totals.newVisits']
label = ['date','visitStartTime']
onehot = ['device.deviceCategory','trafficSource.adwordsClickInfo.adNetworkType',
         'trafficSource.adwordsClickInfo.slot']
hashing = ['channelGrouping','geoNetwork.continent','device.browser','device.operatingSystem','geoNetwork.city','geoNetwork.country',
          'geoNetwork.metro','geoNetwork.networkDomain','geoNetwork.region','geoNetwork.subContinent',
          'trafficSource.adContent','trafficSource.adwordsClickInfo.gclId','trafficSource.keyword',
          'trafficSource.referralPath','trafficSource.source','trafficSource.campaign','trafficSource.medium']
copy2_train_df = copy_train_df.copy()
copy2_train_df.describe(include='all').T
# Check the skew of all numerical features
# skewed_feats = copy2_train_df[num].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness
# skewness = skewness[abs(skewness) > 7]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     #all_data[feat] += 1
#     copy2_train_df[feat] = boxcox1p(copy2_train_df[feat], lam)
# skewed_feats = copy2_train_df[num].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness
# one_zero
def switch1(ts):
    if ts == True:
        return 1
    else:
        return 0
copy2_train_df['device.isMobile'] = copy2_train_df['device.isMobile'].apply(switch1)
# copy2_train_df['device.isMobile'] = copy2_train_df['device.isMobile'].astype(str)

def switch2(ts):
    if ts == 'missing':
        return 0
    else:
        return 1
copy2_train_df['trafficSource.adwordsClickInfo.isVideoAd'] = copy2_train_df['trafficSource.adwordsClickInfo.isVideoAd'].apply(switch2)
# copy2_train_df['trafficSource.adwordsClickInfo.isVideoAd'] = copy2_train_df['trafficSource.adwordsClickInfo.isVideoAd'].astype(str)
copy2_train_df['trafficSource.isTrueDirect'] = copy2_train_df['trafficSource.isTrueDirect'].apply(switch2)
# copy2_train_df['trafficSource.isTrueDirect'] = copy2_train_df['trafficSource.isTrueDirect'].astype(str)
#label
# np.unique(copy2_train_df['date'])
date_ord_map = {'01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6,'07': 7,'08': 8,
               '09': 9,'10': 10,'11': 11,'12': 12}
copy2_train_df['date'] = copy2_train_df['date'].map(date_ord_map)

# np.unique(copy2_train_df['visitStartTime'])
time_ord_map = {'00':0,'01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6,'07': 7,'08': 8,
               '09': 9,'10': 10,'11': 11,'12': 12,'13': 13, '14': 14, '15': 15, '16': 16,
                '17': 17, '18': 18,'19': 19,'20': 20,'21': 21,'22': 22,'23': 23}
copy2_train_df['visitStartTime'] = copy2_train_df['visitStartTime'].map(time_ord_map)
copy3 = copy2_train_df.copy()
# copy3.describe(include='all').T
#onehot
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# transform and map pokemon generations
lbe = LabelEncoder()
ohe = OneHotEncoder()
copy3_ohe = copy3.copy()
# copy3_ohe.drop(onehot, axis=1)
for col in onehot:
    labels = lbe.fit_transform(copy3[col])
    copy3_ohe[col] = labels
# encode generation labels using one-hot encoding scheme
    feature_arr = ohe.fit_transform(
                                  copy3_ohe[[col]]).toarray()
    feature_labels = list(str(col)+'_'+str(cls_label) for cls_label in lbe.classes_)
    features = pd.DataFrame(feature_arr,columns=feature_labels)
    copy3_ohe = pd.concat([copy3_ohe, features], axis=1)

copy3_ohe = copy3_ohe.drop(onehot,axis=1)
# copy3_ohe.describe(include='all').T
# copy3_ohe.columns
copy_ohe = copy3_ohe.copy()
from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=5, input_type='string')
for col in hashing:
    hashed_arr = fh.fit_transform(copy_ohe[col]).toarray()
#     hashed_features = hashed_features.toarray()
    feature_labels = list(str(col)+'_'+str(i) for i in range(5))
    hashed_features = pd.DataFrame(hashed_arr,columns=feature_labels)
    copy_ohe = pd.concat([copy_ohe, hashed_features], axis=1)
copy_ohe.describe(include='all').T
train = copy_ohe.drop(hashing,axis =1)
train.to_csv('encodedData.csv')
train.describe(include='all').T
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,f_regression
drops = ['isRevenue','totals.transactionRevenue','fullVisitorId']
X = train.drop(drops, axis=1)
# X.head()
# X.shape

y_cla = train['isRevenue']
y_reg = train['totals.transactionRevenue']

selector_cla = SelectKBest(f_classif, k=40)
X_cla = selector_cla.fit(X, y_cla)
cols_cla_index = selector_cla.get_support(indices=True)
new_features = X.columns[cols_cla_index]
cols_cla_new = X[new_features]



selector_reg = SelectKBest(f_regression, k=40)
X_reg = selector_reg.fit(X, y_reg)
cols_reg_index = selector_reg.get_support(indices=True)
new_features = X.columns[cols_reg_index]
cols_reg_new = X[new_features]
cla_features = list(X.columns[cols_cla_index])
cla_features.append('fullVisitorId')
cla_features.append('isRevenue')
cla_train = train[cla_features]
reg_features = list(X.columns[cols_reg_index])
reg_features.append('fullVisitorId')
reg_features.append('totals.transactionRevenue')
reg_train = train[reg_features]
cla_train.to_csv('cla_train.csv')
reg_train.to_csv('reg_train.csv')