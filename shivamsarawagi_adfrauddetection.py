import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import gc
from sklearn.ensemble import RandomForestClassifier
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
train = pd.read_csv("../input/train.csv",skiprows=range(1,123903891), nrows=61000000,usecols=train_columns, dtype=dtypes)
yTrain = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)
def timeFeatures(df):
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow']      = df['datetime'].dt.dayofweek
    df["doy"]      = df["datetime"].dt.dayofyear
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df
ip_count = train.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
train = pd.merge(train, ip_count, on='ip', how='left', sort=False)
train.drop(['ip'], axis=1, inplace=True)
del ip_count
gc.collect()
train = timeFeatures(train)
train.head()
model = RandomForestClassifier()
model.fit(train,yTrain)
model.score(train,yTrain)
del train,yTrain
gc.collect()
test = pd.read_csv("../input/test_supplement.csv",usecols=test_columns, dtype=dtypes)
ip_count = test.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
test = pd.merge(test, ip_count, on='ip', how='left', sort=False)
test.drop(['ip'], axis=1, inplace=True)
del ip_count
gc.collect()
test = timeFeatures(test)
test.head()
features = ["app","device","os","channel","clicks_by_ip","dow","doy"]
xTest = test[features]
pred = model.predict(xTest)
del xTest
gc.collect()
my_submission = pd.DataFrame({'click_id': test.click_id, 'is_attributed': pred})
my_submission.to_csv('submission.csv', index=False)