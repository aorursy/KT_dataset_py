import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import csv

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import time

import lightgbm as lgb

import pandas_profiling 

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

color = sns.color_palette()

%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from pylab import rcParams

rcParams['figure.figsize'] = 20, 10

rcParams['font.size'] = 20
training_data = pd.read_csv('../input/save-my-data/train-flattened.csv')

train_row,train_column = training_data.shape

print("The train dataset include {} of rows".format(train_row) + " and {} of columns".format(train_column))

test_data = pd.read_csv('../input/save-my-data/test-flattened.csv')

test_row,test_column = test_data.shape

print("The test dataset include {} of rows".format(test_row) + " and {} of columns".format(test_column))

sub_data = pd.read_csv('../input/save-my-data/submit-flattened.csv')

sub_row,sub_column = sub_data.shape

print("The submit dataset include {} of rows".format(sub_row) + " and {} of columns".format(sub_column))
train_id = training_data.groupby(['fullVisitorId']).size()

test_id = test_data.groupby(['fullVisitorId']).size()

sub_id = sub_data.groupby(['fullVisitorId']).size()

print("{} of common visitors from train to test dataset set".format(len(set(training_data.fullVisitorId.unique()).intersection(set(test_data.fullVisitorId.unique())) )))

print("{} of common visitors from test to submission dataset set".format(len(set(test_data.fullVisitorId.unique()).intersection(set(sub_data.fullVisitorId.unique())) )))

print("{} of common visitors from train to submission dataset set".format(len(set(training_data.fullVisitorId.unique()).intersection(set(sub_data.fullVisitorId.unique())) )))

data = pd.DataFrame({'data_file':['train', 'test', 'submission'], 'user_numbers':[len(train_id), len(test_id), len(sub_id)]})

ax = data.plot.bar(x='data_file', y='user_numbers', rot=0, fontsize = 15, color = "#ff4500",alpha = 0.7)

plt.xlabel('Data Files',fontsize = 10)

plt.ylabel('Number of users',fontsize = 10)
plt.style.use('fivethirtyeight')

tips = []

tips = pd.DataFrame(tips)

tips['train_data_types'] = training_data.dtypes.value_counts()

tips['test_data_types'] = test_data.dtypes.value_counts()

tips.plot.barh(stacked=True,alpha = 0.9)

plt.xlabel("Value Count",fontsize = 20)

plt.ylabel("Data Types",fontsize = 20)

plt.title("Datatypes Diaplay",fontsize = 20)

print('Checking different type of data from train to test dataset.')
training_data.date = pd.to_datetime(training_data.date, format="%Y-%m-%d")

test_data.date = pd.to_datetime(test_data.date, format="%Y-%m-%d")

training_data.date.value_counts().sort_index().plot(label="train", color = "#48D1CC")

test_data.date.value_counts().sort_index().plot(label="test", color = "#ff4500")

plt.legend()

print('It shows the date continous from train and test dataset.')
#split the datetimes for analysis

from datetime import datetime

training_data.date = pd.to_datetime(training_data.date, format = "%Y-%m-%d")

times = []

times = pd.DataFrame(times)

times['date'] = training_data.date

times['weekday'] = times['date'].dt.weekday #extracting week day

times['day'] = times['date'].dt.day # extracting day

times['month'] = times['date'].dt.month # extracting day

times['year'] = times['date'].dt.year

times['visit_Hour'] = (training_data['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
times.head()
times.corr()
training_data['totals_totalTransactionRevenue'] = training_data['totals_totalTransactionRevenue'].astype(float)

group_id = training_data.groupby("fullVisitorId")["totals_totalTransactionRevenue"].sum().reset_index()

f, axarr = plt.subplots(1, 2, figsize=(30, 15) )

axarr[0].plot(group_id['totals_totalTransactionRevenue'])

axarr[1].plot(np.sort(np.log1p(group_id['totals_totalTransactionRevenue'])))

print('We using the value after logrithm to do the predict for the further data')
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)

ax.set_title("logarithm values of totals of transaction revenue", fontsize=20)

ax = sns.distplot(np.log(training_data[training_data['totals_transactionRevenue'] > 0]["totals_transactionRevenue"] + 0.01), bins=40, kde=True)

plt.subplot(1,2,2)

ax = sns.distplot(np.log(training_data[training_data['totals_totalTransactionRevenue'] > 0]["totals_totalTransactionRevenue"] + 0.01), bins=40, kde=True)

print('As the graph shows, the distribution values of totalTransactionRevenue is better than transactionRevenue')
train = training_data.nunique().sort_values()

sns.barplot(x = train.tail(7), y = train.tail(7).index)

plt.xlabel('Value Count',fontsize = 20)

plt.ylabel('Column Name',fontsize = 20)

plt.title('Number of unique data in training dataset',fontsize = 20)

print('As the graph shows below, those five features have the most unique values.')
#function to checking most of values for each features

def data_details(name, dataset):

    name = name.reset_index()

    name['Name'] = name['index']

    name = name[['Name','dtypes']]

    name['Missing_value'] = dataset.isnull().sum().values

    name['zero_value'] = dataset.isin([0]).sum().values

    name['Uniques'] = dataset.nunique().values

    return name
rest_data = training_data[['channelGrouping','date','fullVisitorId']]

rest_data_information = pd.DataFrame(rest_data.dtypes,columns = ['dtypes'])

rest_data_information = data_details(rest_data_information, rest_data)

print("The time period for this searching is {}".format(rest_data[['date']].nunique().values))

rest_data_information
channel_types = rest_data.groupby(['channelGrouping']).size()

sns.barplot(x = channel_types, y = channel_types.index)

plt.xlabel('Value Count',fontsize = 30)

plt.ylabel('Channel_Type Name',fontsize = 30)

plt.title('Types of channels in training dataset',fontsize = 30)

print('Checking all the channels from training dataset:')
#device part

device = training_data.filter(like = 'device')

device_information = pd.DataFrame(device.dtypes,columns = ['dtypes'])

device_information = data_details(device_information, device)

device_information
os = device.groupby(['device_browser']).size().sort_values()

os = os.tail(5)

plt.figure(figsize=(13,6))

bt = sns.boxenplot(x='device_browser', y='totals_totalTransactionRevenue', 

                   data=training_data[(training_data['device_browser'].isin((device.groupby(['device_browser']).size().sort_values().tail(5).index.values))) &

                                  np.log1p(training_data['totals_totalTransactionRevenue']).dropna() > 0])

bt.set_title('Top 5 Browsers Name by Transactions Revenue', fontsize=45)

bt.set_xticklabels(bt.get_xticklabels(),rotation=20)

bt.set_xlabel('Device Names', fontsize=18)

bt.set_ylabel('Trans Revenue Dist', fontsize=18)

plt.show()
deviceCategory = device.groupby(['device_deviceCategory']).size()

sns.barplot(x = deviceCategory, y = deviceCategory.index)

plt.xlabel('Value Count',fontsize = 30)

plt.ylabel('device Category Name',fontsize = 30)

plt.title('Device Categories in training dataset',fontsize = 30)

print('The main of the device categories include Desktop, Mobel, Tablet:')
Mobile = device.groupby(['device_isMobile']).size()

sns.barplot(x = Mobile.index, y = Mobile)

plt.xlabel('Mobile user or not',fontsize = 30)

plt.ylabel('Number of the user',fontsize = 30)

plt.title('Mobile using in training dataset',fontsize = 30)

print('As the result shows, most of users are not using the mobile:')
rcParams['font.size'] = 20

rcParams['figure.figsize'] = 15, 10

operatingSystem = device.groupby(['device_operatingSystem']).size().sort_values()

sns.barplot(x = operatingSystem.tail(8).index, y = operatingSystem.tail(8))

plt.xlabel('Opterating Systems',fontsize = 30)

plt.ylabel('Value Count',fontsize = 30)

plt.title('Operating System Diaplay',fontsize = 30)

print('As the graph shows, the most populer system is Windows.')
geoNetwork = training_data.filter(like = 'geoNetwork')

geoNetwork_information = pd.DataFrame(geoNetwork.dtypes,columns = ['dtypes'])

geoNetwork_information = data_details(geoNetwork_information, geoNetwork)

geoNetwork_information
import squarify

import random

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(20)]

country_name = geoNetwork['geoNetwork_country'].value_counts()

country_name = round((geoNetwork['geoNetwork_country'].value_counts()[:30] \

                       / len(geoNetwork['geoNetwork_country']) * 100),2)

top_country = squarify.plot(sizes=country_name.values, label=country_name.index,value=country_name.values,alpha=.4, color=color)

top_country.set_title("'TOP 30 Countrys - % size of total",fontsize=20)

top_country.set_axis_off()

plt.figure(figsize=(2,10))

plt.show()
country_region = geoNetwork['geoNetwork_region'].value_counts()

country_region = round((geoNetwork['geoNetwork_region'].value_counts()[:30] \

                       / len(geoNetwork['geoNetwork_region']) * 100),2)

top_region = squarify.plot(sizes=country_region.values, label=country_region.index,value=country_region.values,alpha=.4, color=color)

top_region.set_title("'TOP 30 Regions - % size of total",fontsize=20)

top_region.set_axis_off()

plt.figure(figsize=(2,15))

plt.show()
totals = training_data.filter(like = 'totals')

totals_information = pd.DataFrame(totals.dtypes,columns = ['dtypes'])

totals_information = data_details(totals_information, totals)

totals_information
df = totals

plt.figure(figsize=(10,10), dpi= 80)

sns.heatmap(df.corr(),xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

plt.title('Correlogram about Total Transaction Revenue', fontsize=22)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()

print('As this graph shows, totals_newVisits, totals_pageviews, total_sessionQualityDim, totals_timeOnSite shows closer relationships.')
f, axarr = plt.subplots(1, 3, figsize=(30, 5))

axarr[0].plot(totals['totals_hits'])

axarr[1].plot(totals['totals_pageviews'].dropna())

axarr[2].plot(totals['totals_sessionQualityDim'].dropna())

print('As the graph shows below: the standerdiviation values of the totals_sessionQualityDim is lower than other two columns of the data')
f, axarr = plt.subplots(1, 3, figsize=(30, 5) )

axarr[0].plot(totals['totals_timeOnSite'].dropna())

axarr[1].plot(totals['totals_transactionRevenue'].dropna())

axarr[2].plot(totals['totals_transactions'].dropna())
trafficSource = training_data.filter(like = 'trafficSource')

trafficSource_information = pd.DataFrame(trafficSource.dtypes,columns = ['dtypes'])

trafficSource_information = data_details(trafficSource_information, trafficSource)

trafficSource_information
rcParams['figure.figsize'] =15, 20

rcParams['font.size'] = 20

adContent = training_data.groupby(['trafficSource_adContent']).size().sort_values()

sns.barplot(x = adContent.tail(23), y = adContent.tail(23).index)

plt.xlabel('Value Count',fontsize = 30)

plt.ylabel('Ad content types',fontsize = 30)

plt.title('Operating System Diaplay',fontsize = 30)

print('After tidying the data, 23 of the unique values shows more than one time:')
visit = training_data.filter(like = 'visit')

visit_information = pd.DataFrame(visit.dtypes,columns = ['dtypes'])

visit_information = data_details(visit_information, visit)

visit_information
df1 = visit

plt.figure(figsize=(6,6), dpi= 80)

sns.heatmap(df1.corr(),xticklabels=df1.corr().columns, yticklabels=df1.corr().columns, cmap='RdYlGn', center=0, annot=True)

plt.title('Correlogram of Total Transaction Revenue', fontsize=22)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
plt.figure(figsize=(20,6))

(training_data.visitStartTime == training_data.visitId).value_counts()

(training_data.visitStartTime == training_data.visitId).value_counts().plot.barh(alpha = 0.8,color = ["orange","pink"])

print('It shows most of the visitId edit base on the visit start times.')
#For constant values (Those feature mate shows like not avaliable datasets)

constant_cols = [n for n in training_data.columns if training_data[n].nunique(dropna=False)==1]

print('The constant value of the columns are {}'.format(constant_cols))
train = pd.read_csv('/kaggle/input/google-feature-engineering/concat_train.csv')

test = pd.read_csv('/kaggle/input/google-feature-engineering/test.csv')
train.isMobile = train.isMobile.fillna(-999).apply(lambda x: 1 if x== True else 0)

test.isMobile = test.isMobile.fillna(-999).apply(lambda x: 1 if x== True else 0)  

train.iloc[:,:-4] = train.iloc[:,:-4].fillna(-999)

Xreturn = train.iloc[:,:-4]

yreturn = train['return']
X_train1, X_test1, y_train1, y_test1 = train_test_split(Xreturn,yreturn, test_size=0.33, random_state=42)

lgb_train1 = lgb.Dataset(X_train1, y_train1)

lgb_test1 = lgb.Dataset(X_test1, y_test1, reference=lgb_train1)
params_return={'learning_rate': 0.1,

 'boosting_type': 'gbdt',

 "metric" : "auc",      

 'max_depth': -1,

 'min_child_weight': 0.001,

 'min_child_samples': 119,

 'reg_alpha': 0,

 'reg_lambda': 1,

 'subsample': 1,

 'colsample_bytree': 1,

 'feature_fraction': 0.5,

 'n_job': 4,

 'random_state':42}
evals_result1 = {}

a1 = lgb.train(params_return, lgb_train1,500,valid_sets=[lgb_train1, lgb_test1], evals_result=evals_result1,early_stopping_rounds=50)
params_return['num_iteration']=a1.best_iteration

ax = lgb.plot_metric(evals_result1, metric='auc')

plt.show()
lgb.plot_importance(a1,max_num_features=10)
from sklearn.decomposition import PCA

graph_score = []

graph_score = pd.DataFrame(graph_score)

pca = PCA(n_components=1)

pca = pca.fit(X_train1,y_train1)

pca_test_value = pca.transform(X_test1)

pca_test_value = pd.DataFrame(pca_test_value)

pca_test_value['PCA_VALUES']= pca_test_value

pca_test_value.hist(grid = False,column = "PCA_VALUES", alpha = 0.5,figsize = (14,5))
ytarget=np.log1p(train['target'][train['return']>0])

Xtarget = train.iloc[:,:-4][train['return']>0]

X_train1, X_test1, y_train1, y_test1 = train_test_split(Xtarget,ytarget, test_size=0.33, random_state=42)

lgb_train1 = lgb.Dataset(X_train1, y_train1)

lgb_test1 = lgb.Dataset(X_test1, y_test1, reference=lgb_train1)
params_regression = {

        'learning_rate': 0.1,

        "objective" : "regression",

        "metric" : "rmse", 

        "max_leaves": 200,

        "num_leaves" : 9,

        "min_child_samples" : 1,

        "learning_rate" : 0.1,

        "bagging_fraction" : 0.9,

        "feature_fraction" : 0.8,

        "bagging_frequency" : 1      

    }
evals_result2 = {}

a2 = lgb.train(params_regression, lgb_train1,500,valid_sets=[lgb_train1, lgb_test1], evals_result=evals_result2,early_stopping_rounds=50)

params_return['num_iteration']=a2.best_iteration

ax = lgb.plot_metric(evals_result2, metric='rmse')

plt.show()
lgb.plot_importance(a2,max_num_features=10)
def get_groupby_target(df,k=0): 

    df['fullVisitorId'] = df['fullVisitorId'].astype(str)

    start_date=(df['date'].min()+ timedelta(days=k))

    end_date=(df['date'].min() + timedelta(days=168+k))

    mask =  (df['date'] > start_date) & (df['date'] <= end_date)

    tf=df.loc[mask]

    tfg = tf.groupby('fullVisitorId').agg({

            'geoNetwork_networkDomain': {'networkDomain': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_city': {'city':lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'device_operatingSystem': {'operatingSystem': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_metro': {'metro': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_region': {'region':lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'channelGrouping': {'channelGrouping':lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_referralPath': {'referralPath': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_country': {'country': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_source': {'source': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_medium': {'medium': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_keyword': {'keyword': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'device_browser':  {'browser': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'trafficSource_adwordsClickInfo.gclId': {'gclId': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'device_deviceCategory': {'deviceCategory': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'geoNetwork_continent': {'continent': lambda x: -999 if pd.isnull(x.dropna().max()) else 1},

            'totals_timeOnSite': {'timeOnSite_max': lambda x: x.dropna().max(),

                                  'timeOnSite_min': lambda x: x.dropna().min(), 

                                  'timeOnSite_mean': lambda x: x.dropna().mean(),

                                  'timeOnSite_sum': lambda x: x.dropna().sum()},

            'totals_pageviews': {'pageviews_max': lambda x: x.dropna().max(),

                                 'pageviews_min': lambda x: x.dropna().min(),

                                 'pageviews_mean': lambda x: x.dropna().mean(),

                                 'pageviews_sum': lambda x: x.dropna().sum()},

            'totals_hits': {'hits_max': lambda x: x.dropna().max(), 

                            'hits_min': lambda x: x.dropna().min(),

                            'hits_mean': lambda x: x.dropna().mean(),

                            'hits_sum': lambda x: x.dropna().sum()},

            'visitStartTime': {'visitStartTime_counts': lambda x: x.dropna().count()},

            'totals_sessionQualityDim': {'sessionQualityDim': lambda x: x.dropna().max()},

            'device_isMobile': {'isMobile': lambda x: x.dropna().max()},

            'visitNumber': {'visitNumber_max' : lambda x: x.dropna().max()}, 

            'totals_totalTransactionRevenue':  {'totalTransactionRevenue_sum':  lambda x:x.dropna().sum()},

            'totals_transactions' : {'transactions' : lambda x:x.dropna().sum()},

            'date':{"session":lambda x :(x.dropna().max()- x.dropna().min()).days}})

    

    target_strat=(start_date+ timedelta(days=214))

    target_end=(target_strat + timedelta(days=62))

    mask1 =  (df['date'] >= target_strat) & (df['date'] <= target_end)

    tf2=df.loc[mask1]

    tf3=tf2.groupby('fullVisitorId').agg({'totals_totalTransactionRevenue':{'target': lambda x: x.dropna().sum()}})

    tfg.columns=tfg.columns.droplevel()

    tf3.columns=['target']

    tfg['fullVisitorId'] = tfg.index

    tf3['fullVisitorId'] = tf3.index

    

    tfg['target']=0

    tfg['return']=0

    for i in tfg['fullVisitorId']:

        if i in tf3['fullVisitorId']:

            tfg.loc[i,'target']=tf3.loc[i,'target']

            tfg.loc[i,'return']=1

    

    return tfg