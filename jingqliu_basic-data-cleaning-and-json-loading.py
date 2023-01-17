import pandas as pd
import json
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_device = pd.DataFrame(list(train.device.apply(json.loads)))
train_geonetwork = pd.DataFrame(list(train.geoNetwork.apply(json.loads)))
train_totals = pd.DataFrame(list(train.totals.apply(json.loads)))
train_trafficSource = pd.DataFrame(list(train.trafficSource.apply(json.loads)))
train_adwordclick = pd.DataFrame(list(train_trafficSource['adwordsClickInfo'].apply(json.dumps).apply(json.loads)))
train_trafficSource = train_trafficSource.drop(['adwordsClickInfo'],axis = 1)
test_device = pd.DataFrame(list(test.device.apply(json.loads)))
test_geonetwork = pd.DataFrame(list(test.geoNetwork.apply(json.loads)))
test_totals = pd.DataFrame(list(test.totals.apply(json.loads)))
test_trafficSource = pd.DataFrame(list(test.trafficSource.apply(json.loads)))
test_adwordclick = pd.DataFrame(list(test_trafficSource['adwordsClickInfo'].apply(json.dumps).apply(json.loads)))
test_trafficSource = test_trafficSource.drop(['adwordsClickInfo'],axis = 1)
train = pd.concat([train[['channelGrouping','date','fullVisitorId','sessionId','socialEngagementType','visitId','visitNumber','visitStartTime']],train_device,train_geonetwork,train_totals,train_trafficSource,train_adwordclick],axis = 1)
test = pd.concat([test[['channelGrouping','date','fullVisitorId','sessionId','socialEngagementType','visitId','visitNumber','visitStartTime']],test_device,test_geonetwork,test_totals,test_trafficSource,test_adwordclick],axis = 1)
print(train.shape)
print(test.shape)
train = train.drop(['targetingCriteria'],axis = 1)
test = test.drop(['targetingCriteria'],axis = 1)
lst = []
for i in train.columns:
    if train[i].value_counts().shape[0] == 1:
        lst.append(i)
train = train.drop(lst,axis = 1)
lst = []
for i in test.columns:
    if test[i].value_counts().shape[0] == 1:
        lst.append(i)
test = test.drop(lst,axis = 1)
print(train.shape)
print(test.shape)
mapper = {'(not set)': 'missing', 'not available in demo dataset': 'missing', '(not provided)': 'missing', 'unknown.unknown': 'missing','(none)': 'missing'}
train = train.fillna('missing')
test = test.fillna('missing')
for i in train.columns:
    train[i] = train[i].astype(str).replace(mapper)
for i in test.columns:
    test[i] = test[i].astype(str).replace(mapper)
train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)