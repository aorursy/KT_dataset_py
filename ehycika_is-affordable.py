# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from collections import Counter

import matplotlib.pyplot as plt

from sklearn import preprocessing



import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/submission.csv')



drop_list = ['index', '非都市土地使用編定', '編號']



train.drop(drop_list, axis=1, inplace=True)

test.drop(drop_list, axis=1, inplace=True)



train = train[train['price_per_ping'] < 3000000]

train = train[train['土地移轉總面積(平方公尺)'] < 1000]

train = train[train['建物移轉總面積(平方公尺)'] < 1000]

train = train[train['車位移轉總面積(平方公尺)'] < 300]

# train = train[train['建物現況格局-廳'] < 10]

train = train[train['建物現況格局-房'] < 40]

train = train[train['建物現況格局-衛'] < 20]



train.reset_index(drop=True, inplace=True)



numeric_cols = [col for col in train.columns if train[col].dtype in [np.float64, np.int64]]
to_be_filled = []



for col in train.columns:

    if col in numeric_cols:

        if train[train[col].isna()].shape[0] > 0:

            to_be_filled.append(col)

            

print(to_be_filled)
def fill_it(cln):

    train[cln] = train.groupby('鄉鎮市區')[cln].transform(lambda x: x.fillna(x.mean()))

    test[cln] = test.groupby('鄉鎮市區')[cln].transform(lambda x: x.fillna(x.mean()))

#     train[cln].fillna(s, inplace=True)

#     test[cln].fillna(t, inplace=True)

    

fill_it('income_avg')

fill_it('income_var')

fill_it('lat')

fill_it('lng')



train['nearest_tarin_station_distance'] = train['nearest_tarin_station_distance'].fillna(train['nearest_tarin_station_distance'].mean())

test['nearest_tarin_station_distance'] = test['nearest_tarin_station_distance'].fillna(test['nearest_tarin_station_distance'].mean())



train['num_of_bus_stations_in_100m'] = train['num_of_bus_stations_in_100m'].fillna(0)

test['num_of_bus_stations_in_100m'] = test['num_of_bus_stations_in_100m'].fillna(0)



train['建築完成年月'] = train['建築完成年月'].fillna(train['建築完成年月'].mean())

test['建築完成年月'] = test['建築完成年月'].fillna(test['建築完成年月'].mean())

def deal_transaction_year(x):

    x = x.astype(str).apply(lambda y: int(y[-18:-15]))

    return x

    

train['transaction_year_feature'] = deal_transaction_year(train['交易年月日'])

test['transaction_year_feature'] = deal_transaction_year(test['交易年月日'])



train = train[train['transaction_year_feature']>90]
# def log1p_it(cols):

#     train[cols] = np.log1p(train[cols])

#     test[cols] = np.log1p(test[cols])

    

# log1p_cols = ['土地移轉總面積(平方公尺)', '建物移轉總面積(平方公尺)', '車位移轉總面積(平方公尺)', 'nearest_tarin_station_distance']

# log1p_it(log1p_cols)
# import seaborn as sb



# def draw_scat(cln):

# #     sb.distplot(train[cln])

#     X = train[cln]

#     Y = train['price_per_ping']

#     print(cln)

#     plt.hist(X)

#     plt.show()

#     plt.scatter(X, Y, c='b', alpha=0.5)

#     plt.show()

    

# cln = [col for col in train.columns if train[col].dtype in [np.float64, np.int64]]



# for col in cln:

#     draw_scat(col)
from sklearn.preprocessing import OneHotEncoder



def cat_it(cln):

    a = train[cln].astype(str).values

    b = test[cln].astype(str).values

    categories = sorted(list(set(a)))

    enc = OneHotEncoder(categories=[categories], handle_unknown='ignore')

    train_onehot = enc.fit_transform(a.reshape(-1,1)).toarray()

    test_onehot = enc.fit_transform(b.reshape(-1,1)).toarray()

    return train_onehot, test_onehot



material_onehot, test_material_onehot = cat_it('主要建材')

usage_onehot, test_usage_onehot = cat_it('主要用途')

transaction_onehot, test_transaction_onehot = cat_it('交易標的')

building_type_onehot, test_building_type_onehot = cat_it('建物型態')

land_usage_cat_onehot, test_land_usage_cat_onehot = cat_it('都市土地使用分區')

dist_onehot, test_dist_onehot = cat_it('鄉鎮市區')

train_station_onehot, test_train_station_onehot = cat_it('nearest_tarin_station')

location_type_onehot, test_location_type_onehot = cat_it('location_type')

park_type_onehot, test_park_type_onehot = cat_it('車位類別')
train['managed'] = train['有無管理組織'].apply(lambda x: 1 if x == '有' else 0)

test['managed'] = test['有無管理組織'].apply(lambda x: 1 if x == '有' else 0)



train['layout'] = train['建物現況格局-隔間'].apply(lambda x: 1 if x == '有' else 0)

test['layout'] = test['建物現況格局-隔間'].apply(lambda x: 1 if x == '有' else 0)

train['land'] = train['交易筆棟數'].apply(lambda s: int(re.findall(r'土地(\d)', s)[0]))

train['build'] = train['交易筆棟數'].apply(lambda s: int(re.findall(r'建物(\d)', s)[0]))

train['park'] = train['交易筆棟數'].apply(lambda s: int(re.findall(r'車位(\d)', s)[0]))



test['land'] = test['交易筆棟數'].apply(lambda s: int(re.findall(r'土地(\d)', s)[0]))

test['build'] = test['交易筆棟數'].apply(lambda s: int(re.findall(r'建物(\d)', s)[0]))

test['park'] = test['交易筆棟數'].apply(lambda s: int(re.findall(r'車位(\d)', s)[0]))
X_columns = [col for col in train.columns if train[col].dtype in [np.float64, np.int64]]

X_columns.remove('price_per_ping')
from  sklearn.model_selection import train_test_split



X = train[X_columns].values

X = np.concatenate(

    [X, material_onehot, usage_onehot, transaction_onehot, building_type_onehot, land_usage_cat_onehot, dist_onehot, train_station_onehot, location_type_onehot, park_type_onehot],

    axis=1

)

Y = train[['price_per_ping']].values



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1212)
import xgboost

from sklearn.metrics import r2_score



gpuconf = { 

    'tree_method': 'gpu_hist',

    'predictor': 'gpu_predictor',

    'max_depth': 10,

    'n_estimators': 1300, #1000

    'learning_rate': 0.02, #0.08

    'gamma': 0,

    'subsample': 0.7,

    'n_jobs': 4,

    'objective': 'reg:linear',

    'alpha': 0.00006

}



cpuconf = { 

    'max_depth': 10,

    'n_estimators': 2000,#2000,

    'learning_rate': 0.01,#0.01,

    'gamma': 0,

    'subsample': 0.7,

    'n_jobs': 4,

    'objective': 'reg:linear',

    'alpha': 0.00006

}



xgb = xgboost.XGBRegressor(**cpuconf)



xgb.fit(X_train, Y_train)

predictions = xgb.predict(X_test)

# print(f'R2 Score: {r2_score(np.expm1(Y_test), np.expm1(predictions))}')

print(f'R2 Score: {r2_score(Y_test, predictions)}')
X = test[X_columns].values

X = np.concatenate(

    [X, test_material_onehot, test_usage_onehot, test_transaction_onehot, test_building_type_onehot, test_land_usage_cat_onehot, test_dist_onehot, test_train_station_onehot, test_location_type_onehot, test_park_type_onehot],

    axis=1

)



predictions = xgb.predict(X)

my_submission = pd.DataFrame({'index':submission.index,'price_per_ping': predictions})

my_submission.to_csv('submission.csv', index=False)

print(predictions)