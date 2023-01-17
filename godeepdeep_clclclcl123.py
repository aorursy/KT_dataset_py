# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold  

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import xgboost as xgb 

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
# debug = True



# if debug:

#     nrows = 50000

# else:

#     nrows = None
# %%time

# kaggle = '../input/'

# train = pd.read_csv(kaggle +'train.csv', nrows = nrows,parse_dates=['Dates'])

# test = pd.read_csv(kaggle + 'test.csv', nrows = nrows, parse_dates=['Dates'], index_col='Id')

# test = test.sample(frac = 0.1)
# %%time

# kaggle = '../input/'

# train = pd.read_csv(kaggle +'train.csv', nrows = nrows,parse_dates=['Dates'])

# test = pd.read_csv(kaggle + 'test.csv', nrows = nrows, parse_dates=['Dates'], index_col='Id')

# test = test.sample(frac = 0.1)import pandas as pd



import numpy as np

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

pd.options.display.max_columns=100

train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')



def feature_engineering(data):

    data['Date'] = pd.to_datetime(data['Dates'].dt.date)

    data['n_days'] = (data['Date'] - data['Date'].min()).apply(lambda x: x.days)

    data['Day'] = data['Dates'].dt.day

    data['DayOfWeek'] = data['Dates'].dt.weekday

    data['Month'] = data['Dates'].dt.month

    data['Year'] = data['Dates'].dt.year

    data['Hour'] = data['Dates'].dt.hour

    data['Minute'] = data['Dates'].dt.minute

    data['Block'] = data['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)

    data["X_Y"] = data["X"] - data["Y"]

    data["XY"] = data["X"] + data["Y"]

    data.drop(columns=['Dates','Date','Address'], inplace=True)

    return data

train = feature_engineering(train)

test = feature_engineering(test)

train.drop(columns=['Descript','Resolution'], inplace=True)


# rare_cats = set(['FAMILY OFFENSES', 'BAD CHECKS', 'BRIBERY', 'EXTORTION',

#        'SEX OFFENSES NON FORCIBLE', 'GAMBLING', 'PORNOGRAPHY/OBSCENE MAT',

#        'TREA'])

# all_cats = set(train['Category'].unique())

# common_cats = all_cats-rare_cats


# train= train[train['Category'].isin(common_cats)]

# train = train.reset_index(drop = True)
for i in train,test:

    data = i

  

    coord = data[['X','Y']]

    pca = PCA(n_components=2)

    pca.fit(coord)



    coord_pca = pca.transform(coord)



    data['coord_pca1'] = coord_pca[:, 0]

    data['coord_pca2'] = coord_pca[:, 1]
train.head()
# inertia_arr = []



# k_range = range(2, 16)



# for k in k_range:

#     kmeans = KMeans(n_clusters=k, random_state=42).fit(coord)

 

#     # Sum of distances of samples to their closest cluster center

#     interia = kmeans.inertia_

#     print ("k:",k, " cost:", interia)

#     inertia_arr.append(interia)

    

# inertia_arr = np.array(inertia_arr)



# plt.plot(k_range, inertia_arr)

# plt.vlines(5, ymin=inertia_arr.min()*0.9999, ymax=inertia_arr.max()*1.0003, linestyles='--', colors='b')

# plt.title('Elbow Method')

# plt.xlabel('Number of clusters')

# plt.ylabel('Inertia');


# for i in train,test:

#     data = i

#     coord = data[['X','Y']]

#     # kmeans for lat, long

#     kmeans = KMeans(n_clusters=5, random_state=42).fit(coord)

#     coord_cluster = kmeans.predict(coord)

#     data['coord_cluster'] = coord_cluster

#     data['coord_cluster'] = data['coord_cluster'].map(lambda x: 'c_' + str(x).rjust(2, '0'))

# v = cat_val

# for f in v:

#     dist_values = train[f].value_counts().shape[0]

#     print('Variable {} has {} distinct values'.format(f, dist_values))
# print('더미 전 트레인 변수의 수'.format(train.shape[1]))

# train = pd.get_dummies(train,columns=v, drop_first=True)

# print('터미 후 트레인 변수의 수 '.format(train.shape[1]))
le1 = LabelEncoder()

train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])

test['PdDistrict'] = le1.transform(test['PdDistrict'])



le2 = LabelEncoder()

X = train.drop(columns=['Category'])

y= le2.fit_transform(train['Category'])
train.info()


# #모둔 조합에 대해서 degree 2로 제곱(중첩되는 것은 날려줌, 그러넫 자신을 제곱하는 항이 있냐 없냐는 F하면 제곱

# #하는데 의미 없다고 생각함),

# #인털엑션하고 나머지 바이어스는 뺀다. #self interaction을 



# v1 = ['X','Y','coord_pca1','coord_pca2']



# poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)



# interactions = pd.DataFrame(data=poly.fit_transform(train[v1]), columns=poly.get_feature_names(v1))

# #poly(곱하는 거에)train[v]를 fit시키고 data로 지정해주고,

# #컬럼을 poly식에서 피쳐 네임을 뽑아준다.



# #

# interactions.drop(v1, axis=1, inplace=True)  # Remove the original columns





# # Concat the interaction variables to the train data

# print('Before creating interactions we have {} variables in train'.format(train.shape[1]))



# #여기서 중요한거는 엑스트라 변수를 어디다가 더해주냐인데 axis가 1(열)로 해줘야지 옆으로 붙음 axis=0 이면 아래로 붙어서

# train = pd.concat([train, interactions], axis=1) 

# print('After creating interactions we have {} variables in train'.format(train.shape[1]))


# #모둔 조합에 대해서 degree 2로 제곱(중첩되는 것은 날려줌, 그러넫 자신을 제곱하는 항이 있냐 없냐는 F하면 제곱

# #하는데 의미 없다고 생각함),

# #인털엑션하고 나머지 바이어스는 뺀다. #self interaction을 



# # v = ['X','Y','coord_pca1','coord_pca2']

# v = ['n_days','Day','Month','Year','Hour','Minute']

# poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)



# interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))

# #poly(곱하는 거에)train[v]를 fit시키고 data로 지정해주고,

# #컬럼을 poly식에서 피쳐 네임을 뽑아준다.



# #

# interactions.drop(v, axis=1, inplace=True)  # Remove the original columns





# # Concat the interaction variables to the train data

# print('Before creating interactions we have {} variables in train'.format(train.shape[1]))



# #여기서 중요한거는 엑스트라 변수를 어디다가 더해주냐인데 axis가 1(열)로 해줘야지 옆으로 붙음 axis=0 이면 아래로 붙어서

# #바보같아질 수 있음 

# train = pd.concat([train, interactions], axis=1) 

# print('After creating interactions we have {} variables in train'.format(train.shape[1]))
train.head()
train_data = lgb.Dataset(X, label=y, categorical_feature=['PdDistrict', ])

params = {'boosting':'gbdt',

          'objective':'multiclass',

          'num_class':39,

          'max_delta_step':0.9,

          'min_data_in_leaf': 20,

          'learning_rate': 0.29,

          'max_bin': 501,

          'num_leaves': 41,

          'verbose' : 1}



bst = lgb.train(params, train_data, 120)

predictions_lgb = bst.predict(test)



train_data = lgb.Dataset(X, label=y, categorical_feature=['PdDistrict', ])

params = {'boosting':'gbdt',

          'objective':'multiclass',

          'num_class':39,

          'max_delta_step':0.9,

          'min_data_in_leaf': 4,

          'learning_rate': 0.29,

          'max_bin': 501,

          'num_leaves': 41,

          'verbose' : 1}



bst = lgb.train(params, train_data, 120)

predictions_lgb1 = bst.predict(test)



predictions_lgb1
predictions_lgb
con2 = (predictions_lgb + predictions_lgb1)/2

con2


# # train_xgb = xgb.DMatrix(X, label=y)

# # test_xgb  = xgb.DMatrix(test)

# dtrain = xgb.DMatrix(X, label=y)

# dtest  = xgb.DMatrix(test) 
# params = {

#     'max_depth': 5, 

#     'eta': 0.3,  

#     'num_boost_rounds' : 150 ,

#     'silent': 1, 

#     'objective': 'multi:softprob',  

#     'eval_metric' : 'mlogloss',

#     'learning_data' : 0.07,

#     'num_class': 39,

#     'min_child_weight':1,

# }
# model_xgb = xgb.train(params, dtrain, 10)

# predictions_xgb = model_xgb.predict(dtest)



#하이퍼 파라미터 테스트의 수행 속도를 향상시키기 위해 n_estimators 를 100으로 감소

# predictions_xgb
# con1 = (predictions_lgb + predictions_xgb)/2

# con1
submission = pd.DataFrame(con2, columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')),

                          index=test.index)

 

submission.to_csv('submission.csv', index='Id')
submission