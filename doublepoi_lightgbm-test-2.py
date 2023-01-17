#作者：1621430024

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import lightgbm as lgb
train_data = pd.read_csv('../input/sf-crime/train.csv.zip', parse_dates=['Dates'])

test_data = pd.read_csv('../input/sf-crime/test.csv.zip', parse_dates=['Dates'])
train_data.info()

test_data.info()
all_features = pd.concat((train_data.iloc[:, [0, 3, 4, 6, 7, 8]],

                          test_data.iloc[:, [1, 2, 3, 4, 5, 6]]),

                         sort=False)



num_train = train_data.shape[0]



train_labels = pd.get_dummies(train_data['Category']).values

num_outputs = train_labels.shape[1]

train_labels = np.argmax(train_labels, axis=1)



all_features['year'] = all_features.Dates.dt.year

all_features['month'] = all_features.Dates.dt.month

all_features['new_year'] = all_features['month'].apply(

    lambda x: 1 if x == 1 or x == 2 else 0)

all_features['day'] = all_features.Dates.dt.day

all_features['hour'] = all_features.Dates.dt.hour

all_features['evening'] = all_features['hour'].apply(lambda x: 1

                                                     if x >= 18 else 0)



wkm = {

    'Monday': 0,

    'Tuesday': 1,

    'Wednesday': 2,

    'Thursday': 3,

    'Friday': 4,

    'Saturday': 5,

    'Sunday': 6

}

all_features['DayOfWeek'] = all_features['DayOfWeek'].apply(lambda x: wkm[x])

all_features['weekend'] = all_features['DayOfWeek'].apply(

    lambda x: 1 if x == 4 or x == 5 else 0)



OneHot_features = pd.get_dummies(all_features['PdDistrict'])



all_features['block'] = all_features['Address'].apply(

    lambda x: 1 if 'block' in x.lower() else 0)



PCA_features = all_features[['X', 'Y']].values

Standard_features = all_features[['DayOfWeek', 'year', 'month', 'day',

                                  'hour']].values

OneHot_features = pd.concat([

    OneHot_features, all_features[['new_year', 'evening', 'weekend', 'block']]

],

                            axis=1).values



scaler = StandardScaler()

scaler.fit(Standard_features)

Standard_features = scaler.transform(Standard_features)



pca = PCA(n_components=2)

pca.fit(PCA_features)

PCA_features = pca.transform(PCA_features)



all_features = np.concatenate(

    (PCA_features, Standard_features, OneHot_features), axis=1)



train_features = all_features[:num_train]

num_inputs = train_features.shape[1]

test_features = all_features[num_train:]
data_train = lgb.Dataset(train_features, label = train_labels)
params = {

    'boosting': 'gbdt', 

    'objective': 'multiclass',

    'metrics' : 'multi_logloss',

    'num_class': num_outputs,

    'verbosity': 1,

    'device_type':'gpu',

    'gpu_platform_id':0,

    'gpu_device_id':0,                #以上不再调整

    'max_depth': 6,

    'num_leaves': 50,                 #常用数值，备调，step2

    'min_data_in_leaf' : 20,          #默认数值，备调，step3

    'feature_fraction': 0.8,          #常用数值，备调，step4

    'learning_rate': 0.1,             #默认数值，备调，step5

    }

gbm = lgb.train(params, data_train, num_boost_round = 214)

gbm.save_model('../working/gbm(v1).txt')

testResult = gbm.predict(test_features)

sampleSubmission = pd.read_csv('../input/sf-crime/sampleSubmission.csv.zip')

Result_pd = pd.DataFrame(testResult,

                         index=sampleSubmission.index,

                         columns=sampleSubmission.columns[1:])

Result_pd.to_csv('../working/sampleSubmission(gbmv1).csv', index_label='Id')