# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
def load_df(csv_path, nrows=None):
    USE_COLUMNS = [
        'channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',
        'socialEngagementType', 'totals', 'trafficSource', 'visitId',
        'visitNumber', 'visitStartTime', 'customDimensions'
        #'hits'
    ]
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows, usecols=USE_COLUMNS)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
train_df = load_df("../input/train_v2.csv")
test_df = load_df("../input/test_v2.csv")
print(train_df.columns.difference(test_df.columns))
print(test_df.columns.difference(train_df.columns))
train_df = train_df.drop(labels=['trafficSource.campaignCode'],axis=1)
train_df = train_df.drop(labels=['customDimensions','totals.totalTransactionRevenue'],axis=1)
test_df = test_df.drop(labels=['customDimensions','totals.transactionRevenue','totals.totalTransactionRevenue'],axis=1)
#Converting the datatype of date field
train_df['date']= pd.to_datetime(train_df['date'],format='%Y%m%d')
test_df['date']= pd.to_datetime(test_df['date'],format='%Y%m%d')
train_df.head()
#We need to predict totals.transactionRevenue. Lets explore that variable
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].astype('float')
print('Total records: ', len(train_df), 'customers doesnt contribute for revenue: ',train_df['totals.transactionRevenue'].isna().sum(), \
      'customers contributing revenue: ',len(train_df) - train_df['totals.transactionRevenue'].isna().sum())

#Lets start with data cleaning
#Find any unique value columns
print('Features with one unique values are :')
print(train_df.columns[train_df.nunique()==1])
new_df = train_df.drop(labels=train_df.columns[train_df.nunique()==1],axis=1)
test_df = test_df.drop(labels=train_df.columns[train_df.nunique()==1],axis=1)
print(new_df.shape)
pd.options.display.max_columns=new_df.shape[1]
print(new_df.info())
new_df.head()

geofields = ['geoNetwork.city','geoNetwork.continent','geoNetwork.country','geoNetwork.metro','geoNetwork.networkDomain','geoNetwork.region','geoNetwork.subContinent']

for fields in geofields:
    temp_df = new_df.groupby(by=fields).size().sort_values(ascending=False).head(10)
    print(temp_df)
    print('*'*30)
    
temp = new_df.groupby(by=['fullVisitorId']).agg({'visitId':'count','totals.transactionRevenue':'sum'})
print(temp.corr())
temp.plot(kind='scatter',x='visitId',y='totals.transactionRevenue')
#Lets analyze how the visits and transactionRevenue trend with date
plot_df = new_df.groupby(by=['date']).agg({'fullVisitorId':'count','totals.transactionRevenue':'sum'}).reset_index()

fig = plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
plt.plot(pd.to_datetime(plot_df['date']),plot_df['totals.transactionRevenue'])
plt.ylabel('Revenue')
plt.xticks(rotation=90)

plt.subplot(2,1,2)
plt.plot(pd.to_datetime(plot_df['date']),plot_df['fullVisitorId'])
plt.ylabel('Visits')
plt.xticks(rotation=90);

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
#Weekly pattern

plot_df = new_df.groupby(by=['date']).agg({'fullVisitorId':'count','totals.transactionRevenue':'sum'})

plot_df = plot_df.resample('W').mean()

plt.figure(figsize=(15,8))
plt.subplot(2,1,1)
plt.plot(plot_df.index,plot_df['totals.transactionRevenue'])
plt.ylabel('Revenue')

plt.subplot(2,1,2)
plt.plot(plot_df.index,plot_df['fullVisitorId'])
plt.ylabel('Visits');
plot_df = new_df.groupby(by=['fullVisitorId']).agg({'channelGrouping':'count','totals.transactionRevenue':'sum'})

plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.hist(np.log1p(plot_df[plot_df['totals.transactionRevenue']>0]['totals.transactionRevenue']))

plt.subplot(2,1,2)
plt.hist(np.log1p(plot_df[plot_df['totals.transactionRevenue']>0]['totals.transactionRevenue']))
#new_df = new_df.drop(labels=['sessionId','visitId','visitStartTime'])
new_df.isna().sum()
new_df['totals.transactionRevenue'].fillna(value=0,inplace=True)
transaction_df = new_df[new_df['totals.transactionRevenue']>0.0]
nontransaction_df = new_df[new_df['totals.transactionRevenue']<=0]
print(transaction_df.shape, nontransaction_df.shape)
#new_df['totals.transactionRevenue']>0.0
def getplot(df1, df2, groupfield):
    fig =plt.figure(figsize=(15,8))
    plt.subplot(2,1,1)
    plot_df = df1.groupby(by=[groupfield])['totals.transactionRevenue'].size().reset_index()
    plot_df = plot_df.sort_values(by=['totals.transactionRevenue'], ascending=False).head(10)
    plt1 = plt.bar(plot_df[groupfield],plot_df['totals.transactionRevenue'])

    plt.subplot(2,1,2)
    plot_df = df2.groupby(by=[groupfield])['totals.transactionRevenue'].size().reset_index()
    plot_df = plot_df.sort_values(by=['totals.transactionRevenue'], ascending=False).head(10)
    plt2 = plt.bar(plot_df[groupfield],plot_df['totals.transactionRevenue'])
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    #fig.tight_layout()
    return plt1, plt2
getplot(transaction_df,nontransaction_df,'geoNetwork.country');
getplot(transaction_df,nontransaction_df,'geoNetwork.city');
getplot(transaction_df,nontransaction_df,'geoNetwork.continent');
getplot(transaction_df,nontransaction_df,'geoNetwork.metro');
getplot(transaction_df,nontransaction_df,'geoNetwork.networkDomain');
getplot(transaction_df,nontransaction_df,'geoNetwork.region');
getplot(transaction_df,nontransaction_df,'geoNetwork.subContinent');
getplot(transaction_df,nontransaction_df,'channelGrouping');
new_df1 = new_df.drop(labels=['geoNetwork.region','geoNetwork.networkDomain','geoNetwork.metro','geoNetwork.continent','geoNetwork.country','visitId','visitStartTime'],axis=1)
test_df1 = test_df.drop(labels=['geoNetwork.region','geoNetwork.networkDomain','geoNetwork.metro','geoNetwork.continent','geoNetwork.country','visitId','visitStartTime'],axis=1)

new_df1 = new_df1.drop(labels=['date','device.isMobile'], axis=1)
test_df1 = test_df1.drop(labels=['date','device.isMobile'], axis=1)
getplot(transaction_df,nontransaction_df,'device.browser');
getplot(transaction_df,nontransaction_df,'device.deviceCategory');
getplot(transaction_df,nontransaction_df,'device.operatingSystem');
transaction_df['trafficSource.adContent'].fillna('NA',inplace=True)
getplot(transaction_df,nontransaction_df,'trafficSource.adContent')
#NA values seems to contribute more for revenue :p
getplot(transaction_df,nontransaction_df,'trafficSource.source')
new_df1 = new_df1.drop(labels=['trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.medium', 'trafficSource.medium', 'trafficSource.referralPath','trafficSource.source'],axis=1)
test_df1 = test_df1.drop(labels=['trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.medium', 'trafficSource.medium', 'trafficSource.referralPath','trafficSource.source'],axis=1)
print(new_df1.shape)
new_df1.head()
new_df1.fillna('0',inplace=True)
test_df1.fillna('0',inplace=True)
new_df1.isna().sum()
new_df1['trafficSource.adContent'].fillna('Noadcontent',inplace=True)
new_df1['trafficSource.keyword'].fillna('NA',inplace=True)
test_df1['trafficSource.adContent'].fillna('Noadcontent',inplace=True)
test_df1['trafficSource.keyword'].fillna('NA',inplace=True)

for columns in ['totals.sessionQualityDim','totals.timeOnSite','totals.transactions']:
    new_df1[columns].fillna('0',inplace=True)
    new_df1[columns] = new_df1[columns].astype('int')
    test_df1[columns].fillna('0',inplace=True)
    test_df1[columns] = test_df1[columns].astype('int')
new_df1.info()
new_df1.head()
def convert_category_todummies(df,field):
    #print('Processing ', field)
    dummy_df = pd.get_dummies(df[field])
    df = pd.concat([df,dummy_df],axis=1)
    df.drop(labels=[field],axis=1,inplace=True)
    return df
def convert_category_tolevel(df,field):
    df[field],index = pd.factorize(df[field])
    return df
#test_df1=test_df1.drop(labels=['totals.transactionRevenue'],axis=1)
#Data cleaning
train_size = new_df1.shape[0]
merged_df = pd.concat([new_df1,test_df1])
merged_df['totals.pageviews']=merged_df['totals.pageviews'].astype('int')
merged_df['totals.hits']=merged_df['totals.pageviews'].astype('int')
merged_df = merged_df.drop(labels=['trafficSource.keyword'],axis=1)
print('Before: ', merged_df.shape)
columns = merged_df.columns
for fields in columns:
    if merged_df[fields].dtype == 'object' and fields not in ['fullVisitorId','method']:
        print('Unique values for ', fields, len(merged_df[fields].unique()), merged_df[fields].unique())
        if len(merged_df[fields].unique()) > 40:
            print('Level conversion')
            merged_df[fields] = convert_category_tolevel(merged_df,fields)
            merged_df[fields] = merged_df[fields].astype('int')
        else:
            print('One hot conversion')
            merged_df = convert_category_todummies(merged_df,fields)
            #merged_df = merged_df.drop(labels=[fields],axis=1)
print('After: ', merged_df.shape)
from sklearn.preprocessing import MinMaxScaler

scaler1 = MinMaxScaler()

#merged_df1 = merged_df.groupby(by=['fullVisitorId']).mean()
merged_df['totals.transactionRevenue'] = np.log1p(merged_df['totals.transactionRevenue'])


new_df1 = merged_df.iloc[:train_size]
test_df1 = merged_df[train_size:]
#new_df1 = new_df1.drop(labels=['method'],axis=1)
test_df1 = test_df1.drop(labels=['totals.transactionRevenue'],axis=1)
print(new_df1.shape, test_df1.shape)
import gc
del train_df, new_df, test_df,merged_df, transaction_df, nontransaction_df, plot_df
gc.collect()
#we need to predict log revenue per customer. Lets group by full visitor id
train_x = new_df1.groupby(by=['fullVisitorId']).mean()
del new_df1
gc.collect()

train_y = train_x['totals.transactionRevenue']
train_x = train_x.drop(labels=['totals.transactionRevenue'],axis=1)
scaled_x = scaler1.fit_transform(train_x.values)
#train_y = np.log1p(train_y)
train_x = pd.DataFrame(scaled_x, columns=train_x.columns)
import lightgbm as lgb
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.15, random_state=1)

lgb_train_data = lgb.Dataset(X_train, label=y_train)
lgb_val_data = lgb.Dataset(X_val, label=y_val)

params = {
        "objective" : "regression",
        "metric" : "rmse",
        #"num_leaves" : 40,
        "max_depth" : 10,
        "boosting" : "gbdt",
        "learning_rate" : 0.0025,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "seed": 42}
model = lgb.train(params, lgb_train_data, 
                      num_boost_round=5000,
                      valid_sets=[lgb_train_data, lgb_val_data],
                      early_stopping_rounds=100,
                      verbose_eval=500)
test_x = test_df1.groupby(by=['fullVisitorId']).mean()
visitorid = test_x.index
scaled_test = scaler1.transform(test_x)
test_x = pd.DataFrame(scaled_test,columns=test_x.columns)
from sklearn.metrics import mean_squared_error
y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
y_pred_submit = model.predict(test_x, num_iteration=model.best_iteration)

print(f"LGBM: RMSE val: {np.sqrt(mean_squared_error(y_val, y_pred_val))}  - RMSE train: {np.sqrt(mean_squared_error(y_train, y_pred_train))}")
plt.style.use('ggplot')
lgb.plot_importance(model,max_num_features=15)
#submission = pd.DataFrame({'fullVisitorId':visitorid,'PredictedLogRevenue':y_pred_submit})
#submission['fullVisitorId']= submission['fullVisitorId'].astype(str)
#submission['PredictedLogRevenue']=submission['PredictedLogRevenue'].apply(lambda x: 0 if x<0 else x)
#submission.to_csv('submission1.csv',index=False)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, Dropout
from keras.callbacks import ReduceLROnPlateau

X_train = X_train.values
X_val = X_val.values
y_train = y_train.values
y_val = y_val.values
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0],1,X_val.shape[1])

#model1 = Sequential()
#model1.add(Bidirectional(LSTM(200,recurrent_dropout=0.2, input_shape=(X_train.shape[1],X_train.shape[2]), kernel_initializer='lecun_normal', return_sequences=True)))
#model1.add(Dropout(0.2))
#model1.add(Bidirectional(LSTM(120,recurrent_dropout=0.2, kernel_initializer='lecun_normal')))
#model1.add(Dropout(0.2))
#model1.add(Dense(50,activation='sigmoid'))
#model1.add(Dropout(0.2))
#model1.add(Dense(20,activation='elu'))
#model1.add(Dense(1,activation='linear'))
#model1.compile(loss='mse', optimizer='adam')


#history = model1.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_val, y_val), verbose=1, shuffle=False)
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()

#test_x = test_x.values
#test_x = test_x.reshape((test_x.shape[0],1,test_x.shape[1]))

#y_pred_train = model1.predict(X_train)
#y_pred_val = model1.predict(X_val)
#y_pred_submit = model1.predict(test_x)

#print(f"LSTM: RMSE val: {np.sqrt(mean_squared_error(y_val, y_pred_val))}  - RMSE train: {np.sqrt(mean_squared_error(y_train, y_pred_train))}")
from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(1,70))
x = Bidirectional(LSTM(200,recurrent_dropout=0.2, kernel_initializer='lecun_normal', return_sequences=True))(inputs)
x = Bidirectional(LSTM(120,recurrent_dropout=0.2, kernel_initializer='lecun_normal'))(x)
x = Dense(50, activation='sigmoid')(x)
x = Dropout(0.1)(x)
x = Dense(20,activation='elu')(x)
output = Dense(1,activation='linear')(x)

model2 = Model(inputs=inputs, outputs=output)
model2.compile(loss='mse', optimizer='adam')
model2.fit(X_train, y_train, epochs=4, batch_size=64, validation_data=(X_val, y_val), verbose=1, shuffle=False)
test_x = test_x.values
test_x = test_x.reshape((test_x.shape[0],1,test_x.shape[1]))

y_pred_train = model2.predict(X_train)
y_pred_val = model2.predict(X_val)
y_pred_submit = model2.predict(test_x)

print(f"LSTM: RMSE val: {np.sqrt(mean_squared_error(y_val, y_pred_val))}  - RMSE train: {np.sqrt(mean_squared_error(y_train, y_pred_train))}")
submission = pd.DataFrame({'fullVisitorId':visitorid,'PredictedLogRevenue':np.squeeze(y_pred_submit)})
submission['fullVisitorId']= submission['fullVisitorId'].astype(str)
submission['PredictedLogRevenue']=submission['PredictedLogRevenue'].apply(lambda x: 0 if x<0 else x)

submission.to_csv('submission2.csv',index=False)
