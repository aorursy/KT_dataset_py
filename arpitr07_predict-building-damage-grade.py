
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
from sklearn.ensemble import ExtraTreesClassifier

train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')
building_csv = pd.read_csv('../input/Building_Structure.csv')
ownership_csv = pd.read_csv('../input/Building_Ownership_Use.csv')

b_id_train = train_csv['building_id']
b_id_test = test_csv['building_id']
b_id_building = building_csv['building_id']
b_id_ownership = ownership_csv['building_id']

damage_train = train_csv['damage_grade']

train_csv = train_csv.drop(['building_id'], axis = 1)
test_csv = test_csv.drop(['building_id'], axis = 1)
building_csv = building_csv.drop(['building_id'], axis = 1)
ownership_csv = ownership_csv.drop(['building_id'], axis = 1)

train_csv = train_csv.drop(['damage_grade'], axis = 1)

train_csv = pd.get_dummies(train_csv)
test_csv = pd.get_dummies(test_csv)
building_csv = pd.get_dummies(building_csv)
ownership_csv = pd.get_dummies(ownership_csv)

building_csv['building_id'] = b_id_building
ownership_csv['building_id'] = b_id_ownership
bsou_csv = pd.merge(building_csv, ownership_csv, on = 'building_id', how = 'inner')
bsou_csv = bsou_csv.drop(['district_id_y', 'vdcmun_id_y', 'ward_id_y'], axis = 1)
bsou_csv.rename(columns = {'district_id_x': 'district_id', 'vdcmun_id_x': 'vdcmun_id', 'ward_id_x': 'ward_id'}, inplace = True)

train_csv['building_id'] = b_id_train
train_csv['damage_grade'] = damage_train
train = pd.merge(train_csv, bsou_csv, on = 'building_id', how = 'inner')
train = train.drop(['district_id_y', 'vdcmun_id_y'], axis = 1)
train.rename(columns = {'district_id_x': 'district_id', 'vdcmun_id_x': 'vdcmun_id'}, inplace = True)

test_csv['building_id'] = b_id_test
test = pd.merge(test_csv, bsou_csv, on = 'building_id', how = 'inner')
test = test.drop(['district_id_y', 'vdcmun_id_y'], axis = 1)
test.rename(columns = {'district_id_x': 'district_id', 'vdcmun_id_x': 'vdcmun_id'}, inplace = True)

def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round((df.isnull().sum()/df.isnull().count()*100), 1).sort_values(ascending = False)
    missing = pd.concat([total, percent], axis = 1, keys = ['Total', '%'])
    return missing

missing_data_train = missing_data(train)
missing_data_train.head(10)

missing_data_test = missing_data(test)
missing_data_test.head(10)

train = train[train['age_building']<=300]

cfpreq = train['count_floors_pre_eq'].values
cfpoeq = train['count_floors_post_eq'].values
train['cfd'] = cfpreq - cfpoeq

hpreq = train['height_ft_pre_eq'].values
hpoeq = train['height_ft_post_eq'].values
train['hd'] = hpreq - hpoeq

train['column_length'] = train['height_ft_pre_eq'].values/train['count_floors_pre_eq'].values

train = train[((train['column_length']<40) & (train['damage_grade'] == "Grade 1")) | ((train['column_length']<40) & (train['damage_grade'] == "Grade 2")) | (train['damage_grade'] == "Grade 3") | (train['damage_grade'] == "Grade 4") | (train['damage_grade'] == "Grade 5")]
train = train[((train['count_floors_pre_eq']<3) & (train['damage_grade'] == "Grade 1")) | ((train['count_floors_pre_eq']<3) & (train['damage_grade'] == "Grade 2")) | ((train['count_floors_pre_eq']<3) & (train['damage_grade'] == "Grade 3")) | ((train['count_floors_pre_eq']<5) & (train['damage_grade'] == "Grade 4")) | ((train['count_floors_pre_eq']<5) & (train['damage_grade'] == "Grade 5"))]
train = train[((train['height_ft_pre_eq']<50) & (train['damage_grade'] == "Grade 1")) | ((train['height_ft_pre_eq']<50) & (train['damage_grade'] == "Grade 2")) | ((train['height_ft_pre_eq']<50) & (train['damage_grade'] == "Grade 3")) | ((train['height_ft_pre_eq']<60) & (train['damage_grade'] == "Grade 4")) | (train['damage_grade'] == "Grade 5")]
train = train[(train['damage_grade'] == "Grade 1") | ((train['count_floors_post_eq']<4) & (train['damage_grade'] == "Grade 2")) | ((train['count_floors_post_eq']<4) & (train['damage_grade'] == "Grade 3")) | ((train['count_floors_post_eq']<4) & (train['damage_grade'] == "Grade 4")) | (train['damage_grade'] == "Grade 5")]

train = train.drop(['height_ft_pre_eq', 'height_ft_post_eq'], axis = 1)
train = train.drop(['count_floors_pre_eq', 'count_floors_post_eq'], axis = 1)

t_cfpreq = test['count_floors_pre_eq'].values
t_cfpoeq = test['count_floors_post_eq'].values
test['cfd'] = np.absolute(t_cfpreq - t_cfpoeq)

t_hpreq = test['height_ft_pre_eq'].values
t_hpoeq = test['height_ft_post_eq'].values
test['hd'] = np.absolute(t_hpreq - t_hpoeq)

test['column_length'] = test['height_ft_pre_eq'].values/test['count_floors_pre_eq'].values
test = test.drop(['height_ft_pre_eq', 'height_ft_post_eq'], axis = 1)
test = test.drop(['count_floors_pre_eq', 'count_floors_post_eq'], axis = 1)

itrain = train.copy()
itest = test.copy()

b_id_itrain = itrain['building_id']
null_itrain = itrain[itrain.isnull().any(axis = 1)]
full_itrain = itrain.dropna(axis = 0)

full_itrain_Y = full_itrain['has_repair_started']
full_itrain_X = full_itrain.drop(['damage_grade', 'building_id', 'has_repair_started'], axis = 1)

iX = full_itrain_X.values
iY = full_itrain_Y.values

imodel = ExtraTreesClassifier()
imodel.fit(iX, iY)

ifeature_importances = pd.Series(list(imodel.feature_importances_))
ifeatures = pd.Series(list(full_itrain_X.columns))
ifi = pd.concat([ifeatures, ifeature_importances], axis = 1, keys = ['Feature', 'Importance']).sort_values(by = ['Importance'], ascending = False)

important_ifeatures = list(ifi.iloc[0:50]['Feature'])

it = full_itrain_X[important_ifeatures]
iXt = it.values
iYt = full_itrain_Y.values

from sklearn.ensemble import RandomForestClassifier
irf = RandomForestClassifier(n_estimators = 300, random_state = 42)
irf.fit(iXt, iYt)

null_itrain_X = null_itrain.drop(['damage_grade', 'building_id', 'has_repair_started'], axis = 1)
null_itrain_X = null_itrain_X.fillna(0)

t_null_itrain = null_itrain_X[important_ifeatures]
t_null_itrain_X = t_null_itrain.values

pred_null_itrain_y = pd.Series(list(irf.predict(t_null_itrain_X)))

b_id_null_itrain = null_itrain['building_id']
b_id_full_itrain = full_itrain['building_id']

null_itrain['has_repair_started'] = pred_null_itrain_y.values

itrain['has_repair_started'] = null_itrain['has_repair_started']

train_f = pd.concat([null_itrain, full_itrain], axis = 0)
missing_data(train_f).head()
train_f = train_f.fillna(0)

train_f_Y = train_f['damage_grade']
train_f_X = train_f.drop(['damage_grade', 'building_id'], axis = 1)
X = train_f_X.values
Y = train_f_Y.values

model = ExtraTreesClassifier()
model.fit(X, Y)

feature_importances = pd.Series(list(model.feature_importances_))
features = pd.Series(list(train_f_X.columns))
fi = pd.concat([features, feature_importances], axis = 1, keys = ['Feature', 'Importance']).sort_values(by = ['Importance'], ascending = False)
fi.head(30)

important_features = list(fi.iloc[0:70]['Feature'])

t = train_f_X[important_features]
Xt = t.values
Yt = train_f_Y.values

test = test.fillna(0)
test_X = test.drop(['building_id'], axis = 1)
t_test = test_X[important_features]
X_t_test = t_test.values

from xgboost import XGBClassifier
xgbc = XGBClassifier(n_estimators=1000, learning_rate=0.2, max_depth=6, random_state=42)
xgbc.fit(Xt, Yt)
pred_test_y = pd.Series(list(xgbc.predict(X_t_test)))

predictions = pd.concat([b_id_test, pred_test_y], axis = 1, keys = ['building_id', 'damage_grade'])

predictions.to_csv('submission.csv',index = False)

