import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime as datetime
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
%matplotlib inline
plt.style.use('ggplot')
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
# Getting Rows and Columns count
train.shape
# First five records
train.head()
# Data information
train.info()
# Statistical Description
train.describe(include="all")
# Show columns
train.columns
# device details
train['device'].head()
# device json columns
device_list=train['device'].apply(json.loads).tolist()
keys=[]

for i in device_list:
    for j in list(i.keys()):
        if j not in keys:
            keys.append(j)
keys
# device json data for first object
train['device'].apply(json.loads)[0]
# Flatten device details and create a new DataFrame for device 
device_df=pd.DataFrame(train['device'].apply(json.loads).tolist())[['browser','deviceCategory','operatingSystem','language','browserVersion','browserSize','mobileDeviceMarketingName','mobileDeviceModel','screenResolution']]
device_df.head()
#geoNetwork data
train['geoNetwork'].head()
# geoNetwork json columns
geoNetwork_list=train['geoNetwork'].apply(json.loads).tolist()
keys=[]
for i in geoNetwork_list:
    for j in list(i.keys()):
        if j not in keys:
            keys.append(j)

keys
# geoNetwork json data for first object
train['geoNetwork'].apply(json.loads)[0]
# Flatten geoNetwork details and create a new DataFrame for geoNetwork 
geoNetwork_df=pd.DataFrame(train['geoNetwork'].apply(json.loads).tolist())[['continent','subContinent','region','country','city','metro','networkLocation','latitude','longitude','networkDomain']]
geoNetwork_df.head()
# totals data
train['totals'].head()
#totals json columns
totals_list=train['totals'].apply(json.loads).tolist()
keys=[]
for i in totals_list:
    for j in list(i.keys()):
        if j not in keys:
            keys.append(j)
            
keys
#totals json data for first object
train['totals'].apply(json.loads)[0]
# Flatten totals details and create a new DataFrame for totals 
totals_df=pd.DataFrame(train['totals'].apply(json.loads).tolist())[['bounces','hits','newVisits','pageviews','visits','transactionRevenue']]
totals_df.head()
# trafficCource data
train['trafficSource'].head()
#trafficSource json columns
trafficSource_list=train['trafficSource'].apply(json.loads).tolist()
keys=[]
for i in trafficSource_list:
    for j in list(i.keys()):
        if j not in keys:
            keys.append(j)
            
keys
#trafficSource json data for first object
train['trafficSource'].apply(json.loads)[0]
# Flatten trafficSource details and create a new DataFrame for trafficSource
trafficSource_df=pd.DataFrame(train['trafficSource'].apply(json.loads).tolist())[['adwordsClickInfo','campaign','keyword','medium','source','isTrueDirect','campaignCode','referralPath','adContent']]
trafficSource_df.head()
#adwordsClickInfo json columns
adwordsClickInfo_list=trafficSource_df['adwordsClickInfo'].apply(json.dumps).apply(json.loads).tolist()
keys=[]
for i in adwordsClickInfo_list:
    for j in i.keys():
        if j not in keys:
            keys.append(j)
            
keys
#adwordsClickInfo json data for first object
trafficSource_df['adwordsClickInfo'].apply(json.dumps).apply(json.loads)[0]
# Flatten adwordsClickInfo details and create a new DataFrame for adwordsClickInfo
adwordsClickInfo_df=pd.DataFrame(trafficSource_df['adwordsClickInfo'].apply(json.dumps).apply(json.loads).tolist())[['criteriaParameters','page','isVideoAd','gclId','adNetworkType','slot','targetingCriteria']]
adwordsClickInfo_df.head()
#Combine the DataFrames
df=pd.concat([train[['channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork','sessionId', 'socialEngagementType', 'totals', 'trafficSource',
       'visitId', 'visitNumber', 'visitStartTime']], device_df, geoNetwork_df, totals_df, trafficSource_df,adwordsClickInfo_df], axis=1)
#Extract Date
df['date']=pd.to_datetime(df['date'], format="%Y%m%d")
df['date'].head()
#Extract Time from visitStartTime timestamp
df['visitStartTime']=pd.to_datetime(df['visitStartTime'],unit='s')
df['visitStartTime'].head()
#Getting year, month and day from date
df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['day']=df['date'].dt.day
df[['year','month','day']].head()
#Getting hour and minute from visitStartTime
df['hour']=df['visitStartTime'].dt.hour
df['minute']=df['visitStartTime'].dt.minute
df['hour'].head()
# First five records
df.head()
# Rows and Columns counts
df.shape
# Show Columns
df.columns
# Data Decsription
# df.describe(include="all")
# Data types
df.dtypes
# Data information
df.info()
# channelGrouping summary
df['channelGrouping'].value_counts()
# Visualize channelGrouping summary
df['channelGrouping'].value_counts().plot(kind='bar', figsize=(15,8))
# Browser summary
df['browser'].value_counts()
# Visualize browser summary
df['browser'].value_counts().head(10).plot(kind='bar', figsize=(15,8))
# deviceCategory summary
df['deviceCategory'].value_counts()
# Visualize deviceCategory summary
df['deviceCategory'].value_counts().plot(kind='bar', figsize=(15,8))
# operatingSystem summary
df['operatingSystem'].value_counts()
# Visualize operatingSystem summary
df['operatingSystem'].value_counts().plot(kind='bar', figsize=(15,8))
# continent summary
df['continent'].value_counts()
# Visualize continent summary
df['continent'].value_counts().plot(kind='bar', figsize=(15,8))
# subContinent summary
df['subContinent'].value_counts()
# Visualize subContinent summary
df['subContinent'].value_counts().plot(kind='bar',figsize=(15,8))
# country summary
df['country'].value_counts()
# Visualize country summary
df['country'].value_counts().head(10).plot(kind='bar', figsize=(15,8))
# region summary
df['region'].value_counts()
# Visualize region summary
df['region'].value_counts().head(10).plot(kind='bar', figsize=(15,8))
# medium summary
df['medium'].value_counts()
# Visualize medium summary
df['medium'].value_counts().plot(kind='bar', figsize=(15,8))
# source summary
df['source'].value_counts()
# Visualize source summary
df['source'].value_counts().head(10).plot(kind='bar', figsize=(15,8))
# year summary
df['year'].value_counts()
 #Visualize year summary
df['year'].value_counts().plot(kind='bar', figsize=(15,8))
# Summary by country and number of hits
df.groupby(['country'])['hits'].count()
# visualizing country and number of hits
df.groupby(['country'])['hits'].count().sort_values().tail(10).plot(kind='barh', figsize=(15,8))
# Summary by country and number of pageviews
df.groupby(['country'])['pageviews'].count()
# visualizing country and number of pageviews
df.groupby(['country'])['pageviews'].count().sort_values().tail(10).plot(kind='barh', figsize=(15,8))
# Doing analysis in transactionRevenue data needs to be in numeric format
df['transactionRevenue']=df['transactionRevenue'].fillna(0)
df['transactionRevenue']=df['transactionRevenue'].astype(np.int64)
df['transactionRevenue'].dtypes
# deviceCategory and total transactionRevenue
df.groupby(['deviceCategory'])['transactionRevenue'].sum()
# visualizing deviceCategory and total transactionRevenue
df.groupby(['deviceCategory'])['transactionRevenue'].sum().plot(kind='bar',figsize=(15,8))
# country and total transactionRevenue
df.groupby(['country'])['transactionRevenue'].sum().sort_values()
# visualizing country and total transactionRevenue
df.groupby(['country'])['transactionRevenue'].sum().sort_values().tail(10).plot(kind='barh', figsize=(15,8))
# browser and total transactionRevenue
df.groupby(['browser'])['transactionRevenue'].sum().sort_values()
# visualizing browser and total transactionRevenue
df.groupby(['browser'])['transactionRevenue'].sum().sort_values().tail(10).plot(kind='barh', figsize=(15,8))
# date and total transactionRevenue
df.groupby(['date'])['transactionRevenue'].sum().sort_values().tail(10)
# visualizing date and total transactionRevenue
df.groupby(['date'])['transactionRevenue'].sum().sort_values().tail(10).plot(kind='barh', figsize=(15,8))
# channelGrouping and total transactionRevenue
df.groupby(['channelGrouping'])['transactionRevenue'].sum()
# visualizing channelGrouping and total transactionRevenue
df.groupby(['channelGrouping'])['transactionRevenue'].sum().sort_values().plot(kind='barh', figsize=(15,8))
df=df[['channelGrouping', 'fullVisitorId', 'visitNumber','browser', 'deviceCategory', 'operatingSystem', 'continent',
       'subContinent', 'region', 'country', 'city', 'bounces', 'hits','newVisits', 'pageviews', 'visits', 'transactionRevenue',
        'campaign', 'keyword', 'medium', 'source','page','adNetworkType','month']]

df.shape
# Fill all NAN values with 0
df=df.fillna(0)
# Data types conversion
df['transactionRevenue']=df['transactionRevenue'].astype(np.int64)
df['hits']=df['hits'].astype(str).astype(np.int)
df['bounces']=df['bounces'].astype(str).astype(np.int)
df['newVisits']=df['newVisits'].astype(str).astype(np.int)
df['pageviews']=df['pageviews'].astype(str).astype(np.int)
df['visits']=df['visits'].astype(str).astype(np.int)
# Transform data
le=preprocessing.LabelEncoder()

df['channelGrouping']=le.fit_transform(df['channelGrouping'].astype(str))
df['browser']=le.fit_transform(df['browser'].astype(str))
df['deviceCategory']=le.fit_transform(df['deviceCategory'].astype(str))
df['operatingSystem']=le.fit_transform(df['operatingSystem'].astype(str))
df['continent']=le.fit_transform(df['continent'].astype(str))
df['subContinent']=le.fit_transform(df['subContinent'].astype(str))
df['region']=le.fit_transform(df['region'].astype(str))
df['country']=le.fit_transform(df['country'].astype(str))
df['city']=le.fit_transform(df['city'].astype(str))
df['campaign']=le.fit_transform(df['campaign'].astype(str))
df['keyword']=le.fit_transform(df['keyword'].astype(str))
df['medium']=le.fit_transform(df['medium'].astype(str))
df['source']=le.fit_transform(df['source'].astype(str))
df['adNetworkType']=le.fit_transform(df['adNetworkType'].astype(str))
df['revenueLog']=df['transactionRevenue'].apply(lambda x :  np.log1p(x) if x>0 else 0)
# Flatten json data
test_device_df=pd.DataFrame(test['device'].apply(json.loads).tolist())[['browser','deviceCategory','operatingSystem','language','browserVersion','browserSize','mobileDeviceMarketingName','mobileDeviceModel','screenResolution']]
test_geoNetwork_df=pd.DataFrame(test['geoNetwork'].apply(json.loads).tolist())[['continent','subContinent','region','country','city','metro','networkLocation','latitude','longitude','networkDomain']]
test_totals_df=pd.DataFrame(test['totals'].apply(json.loads).tolist())[['bounces','hits','newVisits','pageviews','visits']]
test_trafficSource_df=pd.DataFrame(test['trafficSource'].apply(json.loads).tolist())[['adwordsClickInfo','campaign','keyword','medium','source','isTrueDirect','referralPath','adContent']]
test_adwordsClickInfo_df=pd.DataFrame(test_trafficSource_df['adwordsClickInfo'].apply(json.dumps).apply(json.loads).tolist())[['criteriaParameters','page','isVideoAd','gclId','adNetworkType','slot','targetingCriteria']]

# join the flattened dataframes
test_df=pd.concat([test[['channelGrouping', 'date', 'device', 'geoNetwork','sessionId', 'socialEngagementType', 'totals', 'trafficSource',
       'visitId', 'visitNumber', 'visitStartTime']], test_device_df, test_geoNetwork_df, test_totals_df, test_trafficSource_df, test_adwordsClickInfo_df], axis=1)

# transofm date
test_df['date']=pd.to_datetime(test_df['date'], format="%Y%m%d")
test_df['visitStartTime']=pd.to_datetime(test_df['visitStartTime'],unit='s')
#Getting year, month and day from date
test_df['year']=test_df['date'].dt.year
test_df['month']=test_df['date'].dt.month
test_df['day']=test_df['date'].dt.day

#Getting hour and minute from visitStartTime
test_df['hour']=test_df['visitStartTime'].dt.hour
test_df['minute']=test_df['visitStartTime'].dt.minute
test_df=test_df.fillna(0)

# preprocess data
test_df['hits']=test_df['hits'].astype(str).astype(np.int)
test_df['bounces']=test_df['bounces'].astype(str).astype(np.int)
test_df['newVisits']=test_df['newVisits'].astype(str).astype(np.int)
test_df['pageviews']=test_df['pageviews'].astype(str).astype(np.int)
test_df['visits']=test_df['visits'].astype(str).astype(np.int)

test_df['channelGrouping']=le.fit_transform(test_df['channelGrouping'].astype(str))
test_df['browser']=le.fit_transform(test_df['browser'].astype(str))
test_df['deviceCategory']=le.fit_transform(test_df['deviceCategory'].astype(str))
test_df['operatingSystem']=le.fit_transform(test_df['operatingSystem'].astype(str))
test_df['continent']=le.fit_transform(test_df['continent'].astype(str))
test_df['subContinent']=le.fit_transform(test_df['subContinent'].astype(str))
test_df['region']=le.fit_transform(test_df['region'].astype(str))
test_df['country']=le.fit_transform(test_df['country'].astype(str))
test_df['city']=le.fit_transform(test_df['city'].astype(str))
test_df['campaign']=le.fit_transform(test_df['campaign'].astype(str))
test_df['keyword']=le.fit_transform(test_df['keyword'].astype(str))
test_df['medium']=le.fit_transform(test_df['medium'].astype(str))
test_df['source']=le.fit_transform(test_df['source'].astype(str))
test_df['adNetworkType']=le.fit_transform(test_df['adNetworkType'].astype(str))

# Get only useful features as in the training set
test_df=test_df[['channelGrouping', 'visitNumber','browser', 'deviceCategory', 'operatingSystem', 'continent',
       'subContinent', 'region', 'country', 'city', 'bounces', 'hits','newVisits', 'pageviews', 'visits',
        'campaign', 'keyword', 'medium', 'source','page','adNetworkType','month']]

test_df.shape
X=df[['channelGrouping','visitNumber','browser', 'deviceCategory', 'operatingSystem', 'continent',
       'subContinent', 'region', 'country', 'city', 'bounces', 'hits','newVisits', 'pageviews', 'visits',
        'campaign', 'keyword', 'medium', 'source','page','adNetworkType','month']]
y=df['revenueLog']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=7)

rf_model=RandomForestRegressor()
rf_model.fit(X_train,y_train)
pred_test=rf_model.predict(X_test)
print("Sample predictions : ",pred_test[0:5])
print("RMSE : ",np.sqrt(metrics.mean_squared_error(y_test,pred_test)))
importances=rf_model.feature_importances_
indices=np.argsort(importances)[::-1]
features = [X.columns[i] for i in indices]

plt.figure(figsize=(16,9))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), features, rotation=90)
plt.show()
