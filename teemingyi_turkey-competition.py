# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import lightgbm as lgb
from sklearn.model_selection import train_test_split

data = pd.read_json('../input/train.json')
data.head()
data['length_of_embedding'] = data.audio_embedding.apply(len)
print(data.length_of_embedding.describe())

data.length_of_embedding.plot(kind='hist')

data.is_turkey.plot(kind='hist')
pd.crosstab(index=data.length_of_embedding, columns=data.is_turkey)
data['duration'] = data.end_time_seconds_youtube_clip - data.start_time_seconds_youtube_clip
print(data.duration.describe())

data.duration.plot(kind='hist')
pd.crosstab(index=data.length_of_embedding, columns=data.duration)
def create_df(data, i):
    df = pd.DataFrame([x for x in data.audio_embedding.iloc[i]])
    df['is_turkey'] = data.is_turkey.iloc[i]
    return df
vid_df = []
for i in range(len(data.index)):
    vid_df.append(create_df(data, i))
    
len(vid_df)
data_flatten = pd.concat(vid_df)

data_flatten.head()
data_flatten.columns = ['feature_'+str(x) for x in data_flatten.columns[:128]] + ['is_turkey']

print(data_flatten.shape)
data_flatten.head()
# split data into X and y
X = data_flatten.iloc[:,:128]
Y = data_flatten.iloc[:,128]
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
d_train = lgb.Dataset(X_train, label=y_train)

params = {}
params['objective'] = 'binary'

model = lgb.train(params, d_train)
#Prediction
y_pred=model.predict(X_test)
y_pred
# evaluate predictions
from sklearn.metrics import roc_auc_score
mae = roc_auc_score(y_test, y_pred)
print("AUC: {}".format(mae))
d_train_all = lgb.Dataset(X, label=Y)
model_all = lgb.train(params, d_train_all)
data_test = pd.read_json('../input/test.json')

data_test.shape
def create_df_test(data, i):
    df = pd.DataFrame([x for x in data.audio_embedding.iloc[i]])
    df['vid_id'] = data.vid_id.iloc[i]
    return df
vid_df_test = []
for i in range(len(data_test.index)):
    vid_df_test.append(create_df_test(data_test, i))
    
print(len(vid_df_test))

data_flatten_test = pd.concat(vid_df_test)

data_flatten_test.head()
data_flatten_test.columns = ['feature_'+str(x) for x in data_flatten_test.columns[:128]] + ['vid_id']

print(data_flatten_test.shape)
data_flatten_test.head()
#Prediction
test_submit = data_flatten_test.iloc[:,:128]
y_pred_test = model_all.predict(test_submit)
y_pred_test
result = pd.concat([data_flatten_test.vid_id.reset_index(drop=True), pd.Series(y_pred_test, name='is_turkey')], axis=1)
final_result = result.groupby('vid_id').is_turkey.max().sort_values()
final_result = final_result.reset_index()
final_result.to_csv('submission.csv', index=False)
