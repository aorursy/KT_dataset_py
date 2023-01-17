import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.io.json import json_normalize
import json
import gc
%matplotlib inline
json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
json_conv = {col: json.loads for col in (json_cols)}
train = pd.read_csv("../input/train.csv",
                    #nrows = 10000,
                    dtype={'fullVisitorId': str},
                    converters={'device': json.loads,
                               'geoNetwork': json.loads,
                               'totals': json.loads,
                               'trafficSource': json.loads,
                              })
train.head()
train.describe()
train.info()
def extractJsonColumns(df):
    for col in json_cols:
        print('Working on :' + col)
        jsonCol = json_normalize(df[col].tolist())
        jsonCol.columns = [col+'_'+jcol for jcol in jsonCol.columns]
        df = df.merge(jsonCol,left_index=True,right_index=True)
        df.drop(col,axis=1,inplace=True)
    return(df)
train = extractJsonColumns(train)
train.columns
len(train)
def generateColumnInfo(df):
    cls = []
    nullCount = []
    nonNullCount = []
    nullsPct = []
    uniqCount = []
    dataType = []
    for i,col in enumerate(df.columns):
        cls.append(col)
        nullCount.append(df[col].isnull().sum())
        nonNullCount.append(len(df)-df[col].isnull().sum())
        nullsPct.append((df[col].isnull().sum())*(100)/len(df))
        uniqCount.append(df[col].nunique())
        dataType.append(df[col].dtype)
        
    column_info = pd.DataFrame(
        {'ColumnName': cls,
         'NullCount': nullCount,
         'NonNullCount': nonNullCount,
         'NullPercent': nullsPct,
         'UniqueValueCount': uniqCount,
         'DataType':dataType
        })
    return(column_info)
generateColumnInfo(train)

train.drop('trafficSource_campaignCode',axis=1,inplace=True)
trn_colInfo = generateColumnInfo(train)
trn_colInfo[(trn_colInfo['NullCount'] == 0) & (trn_colInfo['UniqueValueCount'] == 1)]
train.drop(['socialEngagementType',
'device_browserSize',
'device_browserVersion',
'device_flashVersion',
'device_language',
'device_mobileDeviceBranding',
'device_mobileDeviceInfo',
'device_mobileDeviceMarketingName',
'device_mobileDeviceModel',
'device_mobileInputSelector',
'device_operatingSystemVersion',
'device_screenColors',
'device_screenResolution',
'geoNetwork_cityId',
'geoNetwork_latitude',
'geoNetwork_longitude',
'geoNetwork_networkLocation',
'totals_visits',
'trafficSource_adwordsClickInfo.criteriaParameters'],axis=1,inplace=True)
plt.figure(figsize=(15,8))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
def plot_colCount(df,col,xtick=0,w=12,h=7):
    plt.figure(figsize=(w,h))
    p = sns.countplot(data =df,x=col)
    plt.xticks(rotation=xtick)
    plt.title('Count of ' + col)
    plt.show()
    
def plot_totalRevenue(df,col,xtick=0,w=12,h=7):
    groupedDf = df.groupby(col,as_index=False)['totals_transactionRevenue'].sum()
    groupedDf = groupedDf[groupedDf['totals_transactionRevenue']>0]
    plt.figure(figsize=(w,h))
    p = sns.barplot(data=groupedDf,x=col,y='totals_transactionRevenue')
    plt.xticks(rotation=xtick)
    plt.title('Total revenue by ' + col)
    plt.show()
    
def plot_revenuePerUnitCol(df,col,xtick=0,w=12,h=7):
    plt.figure(figsize=(w,h))
    plt.ylim()
    p = sns.barplot(data =df,x=col,y='totals_transactionRevenue',ci=False)
    plt.xticks(rotation=xtick)
    plt.title('Revenue per visit')
    plt.show()
print(train['totals_transactionRevenue'].isnull().sum())
train['totals_transactionRevenue'].fillna(0,inplace=True)
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].astype('int64')
plt.figure(figsize=[10,6])
sns.distplot(train['totals_transactionRevenue'])
plot_colCount(train,'channelGrouping',30,10,6)
plot_totalRevenue(train,'channelGrouping',30,10,6)
plot_revenuePerUnitCol(train,'channelGrouping',30,10,6)
train['date'] = pd.to_datetime(train['date'],format='%Y%m%d')
import math
byDate = train.groupby('date',as_index=False).agg({'visitId':'count','totals_transactionRevenue':'sum'}).rename(columns={'visitId':'visits','totals_transactionRevenue':'totalRevenue'})
byDate['totalRevenue'] = byDate['totalRevenue']/1000000
byDateFlat = byDate.melt('date',var_name ='Numbers',value_name='values')
plt.figure(figsize=(16,8))
new_labels = ['label 1', 'label 2']
sns.lineplot(data=byDateFlat,x='date',y='values',hue='Numbers')
plt.title('Visit Count and Total Revenue (in 1000000) by date')
plt.ylabel('')
plt.show()
train['date_year'],train['date_month'],train['date_weekday'] = train['date'].dt.year,train['date'].dt.month,train['date'].dt.weekday
train.drop('date',axis=1,inplace=True)
plot_colCount(train,'date_weekday',0,10,6)
#Monday=0, Sunday=6
plot_totalRevenue(train,'date_weekday',0,10,6)
plot_colCount(train,'date_month',0,10,6)
plot_totalRevenue(train,'date_month',0,10,6)
train['fullVisitorId'].value_counts().head(10)
train.groupby('fullVisitorId').sum()['totals_transactionRevenue'].sort_values(ascending=False).head(10)
train['visitNumber'].value_counts().head()
train['device_browser'].value_counts().head(10)
plot_colCount(train,'device_browser',80)
plot_totalRevenue(train,'device_browser',30,10,6)
plot_revenuePerUnitCol(train,'device_browser',80)
f = sns.FacetGrid(train,hue='device_deviceCategory',size=5,aspect=4)
#plt.xlim(0, 300)
plt.figure(figsize=(15,10))
f.map(sns.kdeplot,'totals_transactionRevenue',shade= True)
f.add_legend()
f = sns.FacetGrid(train,hue='device_deviceCategory',size=5,aspect=4)
plt.xlim(0, 500000000)
plt.figure(figsize=(15,10))
f.map(sns.kdeplot,'totals_transactionRevenue',shade= True)
f.add_legend()
plot_colCount(train,'device_deviceCategory',60)
plot_totalRevenue(train,'device_deviceCategory',30,10,6)
plot_revenuePerUnitCol(train,'device_deviceCategory',60)
plt.figure(figsize=(8,5))
sns.barplot(data =train,x='device_isMobile',y='totals_transactionRevenue')
plot_colCount(train,'device_operatingSystem',60)
plot_totalRevenue(train,'device_operatingSystem',30,10,6)
plot_revenuePerUnitCol(train,'device_operatingSystem',60)
topCities = train['geoNetwork_city'].value_counts().head(50).reset_index()
topCities.columns = ['city','count']
topCities = topCities[topCities.city !='not available in demo dataset']
topCities = topCities[topCities.city !='(not set)']
topCitiesTrain = train[train['geoNetwork_city'].isin(topCities['city'])]
plot_colCount(topCitiesTrain,'geoNetwork_city',60)
plot_totalRevenue(topCitiesTrain,'geoNetwork_city',60)
plot_revenuePerUnitCol(topCitiesTrain,'geoNetwork_city',60)
plot_colCount(train,'geoNetwork_continent',0,10,6)
plot_totalRevenue(train,'geoNetwork_continent',0,10,6)
plot_revenuePerUnitCol(train,'geoNetwork_continent',0,10,6)
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
import plotly.graph_objs as go
import cufflinks as cf
byCountry = train.groupby('geoNetwork_country',as_index=False).agg({'visitId':'count','totals_transactionRevenue':'sum'}).rename(columns={'visitId':'visits','totals_transactionRevenue':'totalRevenue'})
init_notebook_mode(connected=True)
cf.go_offline()
data=dict(type='choropleth',
         locations = byCountry['geoNetwork_country'],
         locationmode = 'country names',
         colorscale = 'Blues',
         reversescale=True,
         text = ['text 1','text 2','text 3'],
         z=byCountry['visits'],
         colorbar={'title':'Total visits'})

layout = dict(title='Visit count by Country')

choromap = go.Figure(data=[data])
iplot(choromap)
data=dict(type='choropleth',
         locations = byCountry['geoNetwork_country'],
         locationmode = 'country names',
         colorscale = 'Blues',
         reversescale=True,
         text = ['text 1','text 2','text 3'],
         z=byCountry['totalRevenue'],
         colorbar={'title':'Total revenue'})

layout = dict(title='Total revenue by Country')

choromap = go.Figure(data=[data])
iplot(choromap)
topCountries = train['geoNetwork_country'].value_counts().head(80).reset_index()
topCountries.columns = ['country','count']
topCountriesTrain = train[train['geoNetwork_country'].isin(topCountries['country'])]
plot_colCount(topCountriesTrain,'geoNetwork_country',80,16)
plot_totalRevenue(topCountriesTrain,'geoNetwork_country',80,16)

plot_revenuePerUnitCol(topCountriesTrain,'geoNetwork_country',80,16)
topCountries = train['geoNetwork_country'].value_counts().head(8).index
def plotByCountry(plotCol,n_labels = 0, xtick=0,plotType = 'line',order=0):
    groupByCountry = train.groupby(['geoNetwork_country',plotCol],as_index=False).count()[['geoNetwork_country',plotCol,'visitId']]
    groupByCountry = groupByCountry[groupByCountry['geoNetwork_country'].isin(topCountries)]
    if n_labels != 0:
        topLabels = train[plotCol].value_counts().head(n_labels).index
        groupByCountry = groupByCountry[groupByCountry[plotCol].isin(topLabels)]
    groupByCountry.columns = ['geoNetwork_country', plotCol, 'visits']
    plt.figure(figsize=[14,10])
    plt.xticks(rotation=xtick)
    if plotType == 'line':
        sns.lineplot(data=groupByCountry,x=plotCol,y='visits',hue='geoNetwork_country')
    if plotType == 'bar':
        if order == 0:
            sns.barplot(data=groupByCountry,x=plotCol,y='visits',hue='geoNetwork_country')
        if order == 1:
            sns.barplot(data=groupByCountry,x='geoNetwork_country',y='visits',hue=plotCol)
plotByCountry(plotCol='date_month',n_labels=12)
plotByCountry('date_weekday')
plotByCountry(plotCol='device_deviceCategory',plotType='bar',order=1)
plotByCountry('channelGrouping',plotType='bar')
plotByCountry('device_operatingSystem',n_labels=5,plotType='bar',order=1)
plotByCountry('device_browser',plotType='bar',n_labels=5,order=1)
topMetrosTrain = train[~train['geoNetwork_metro'].isin(['not available in demo dataset','(not set)'])]
plot_colCount(topMetrosTrain,'geoNetwork_metro',90,16)
plot_totalRevenue(topMetrosTrain,'geoNetwork_metro',90,16)
plot_revenuePerUnitCol(topMetrosTrain,'geoNetwork_metro',90,16)
topRegions = train['geoNetwork_region'].value_counts().head(80).reset_index()
topRegions.columns = ['region','count']
topRegions = topRegions[(topRegions.region !='not available in demo dataset') &(topRegions.region !='(not set)')]
topRegionsTrain = train[train['geoNetwork_region'].isin(topRegions['region'])]
plot_colCount(topRegionsTrain,'geoNetwork_region',80,16)
plot_totalRevenue(topRegionsTrain,'geoNetwork_region',80,16)
plot_revenuePerUnitCol(topRegionsTrain,'geoNetwork_region',80,16)
plot_colCount(train,'geoNetwork_subContinent',30,15,6)
plot_totalRevenue(train,'geoNetwork_subContinent',30,15,6)
plot_revenuePerUnitCol(train,'geoNetwork_subContinent',30,15,6)
train['totals_bounces'].fillna(0,inplace=True)
train['totals_bounces'] = train['totals_bounces'].astype('int64')
train['totals_newVisits'].fillna(0,inplace=True)
train['totals_newVisits'] = train['totals_newVisits'].astype('int64')
train['totals_hits'] = train['totals_hits'].astype('int64')
#totals_pageviews
train['totals_pageviews'] = train['totals_pageviews'].astype(float)
print(train['totals_pageviews'].min())
print(train['totals_pageviews'].max())
train['totals_pageviews'].fillna(0,inplace=True)
sns.distplot(train['totals_pageviews'])
sns.lmplot(data=train,x='totals_pageviews',y='totals_transactionRevenue',
           hue='geoNetwork_continent',col='geoNetwork_continent',col_wrap=2,fit_reg=False)
train['trafficSource_campaign'].value_counts()
train['trafficSource_medium'].value_counts()
train['trafficSource_medium'].replace('(not set)','none',inplace=True)
train['trafficSource_medium'].replace('(none)','none',inplace=True)
plot_colCount(train,'trafficSource_medium',30,10,6)
plot_totalRevenue(train,'trafficSource_medium',30,10,6)
plot_revenuePerUnitCol(train,'trafficSource_medium',30,10,6)
train['trafficSource_source'].value_counts().head()
#trafficSource_adwordsClickInfo.isVideoAd
train['trafficSource_adwordsClickInfo.isVideoAd'].unique()
train.drop(['trafficSource_adwordsClickInfo.isVideoAd'],axis=1,inplace=True)
#trafficSource_isTrueDirect
train['trafficSource_isTrueDirect'].fillna(0,inplace=True)
train['trafficSource_isTrueDirect'].replace(True,1,inplace=True)
train['trafficSource_isTrueDirect']=train['trafficSource_isTrueDirect'].astype(bool)
#trafficSource_adContent
train['trafficSource_adContent'].fillna('Unknown',inplace=True)
#trafficSource_adwordsClickInfo.adNetworkType
train['trafficSource_adwordsClickInfo.adNetworkType'].value_counts()
train['trafficSource_adwordsClickInfo.adNetworkType'].fillna('Unknown',inplace=True)
#trafficSource_adwordsClickInfo.gclId
train['trafficSource_adwordsClickInfo.gclId'].fillna('Unknown',inplace=True)
#trafficSource_adwordsClickInfo.page
train['trafficSource_adwordsClickInfo.page'].fillna(0,inplace=True)
train['trafficSource_adwordsClickInfo.page'] = train['trafficSource_adwordsClickInfo.page'].astype('int64')
#trafficSource_referralPath
train['trafficSource_referralPath'].fillna(0,inplace=True)
#trafficSource_adwordsClickInfo.slot
train['trafficSource_adwordsClickInfo.slot'].value_counts()
train.drop(['trafficSource_adwordsClickInfo.slot'],axis=1,inplace=True)
#trafficSource_keyword
train['trafficSource_keyword'].fillna(0,inplace=True)
train.drop(['sessionId',
            'visitId','visitStartTime',
            'geoNetwork_region'],axis=1,inplace=True)
from sklearn import preprocessing
encoder = preprocessing.OneHotEncoder()
train.columns
unknownLabel = 'zzzUnknown'
leColumns = ['device_deviceCategory','geoNetwork_continent','trafficSource_adwordsClickInfo.adNetworkType',
                'channelGrouping','date_month','date_weekday','geoNetwork_subContinent','trafficSource_medium',
                'geoNetwork_city','geoNetwork_networkDomain','trafficSource_adContent','trafficSource_campaign',
                'trafficSource_keyword','trafficSource_source','device_operatingSystem','device_browser', 
             'geoNetwork_metro','geoNetwork_country','trafficSource_referralPath' ,
             'trafficSource_adwordsClickInfo.gclId']
for col in leColumns:
    print('Processing column ' + col)
    le = preprocessing.LabelEncoder()
    le.fit(train[col].astype(str))
    if unknownLabel not in le.classes_:
        le.classes_ = np.append(le.classes_,unknownLabel)
        #adding unknownLabel to handle test set labels not present in train set
    train[col] = le.transform(train[col].astype(str))
    np.save(col +'.npy',le.classes_)
train.columns
plt.figure(figsize=(26,18))
sns.heatmap(train.corr(),annot=True)
pd.DataFrame(train.corr()['totals_transactionRevenue']).abs().sort_values('totals_transactionRevenue',ascending=False).head(30)
import math
from sklearn.model_selection import train_test_split
X = train.drop(['totals_transactionRevenue','fullVisitorId'],axis=1)
y = train['totals_transactionRevenue'].apply(lambda x:0 if x==0 else math.log(x))    
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train))
print(len(X_val))
import lightgbm as lgb
from math import sqrt
from sklearn.metrics import mean_squared_error

params = {'objective' : 'regression','metric' :'rmse','bagging_fraction' :0.5, 'bagging_frequency':8 ,'feature_fraction':0.7, 'learning_rate':0.01, 'max_bin' :100, 
           'max_depth' :7, 'num_leaves':30}

lgbmReg = lgb.LGBMRegressor(**params,n_estimators=1000) 
lgbmReg.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=100,verbose=30,eval_metric='rmse')
imp = pd.DataFrame({'Feature':X_val.columns,'Importance':lgbmReg.booster_.feature_importance()})
imp.sort_values(by='Importance',ascending=False)
plt.figure(figsize=(14,20))
sns.barplot(data=imp.sort_values(by='Importance',ascending=False),x='Importance',y='Feature')