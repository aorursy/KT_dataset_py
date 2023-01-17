# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# サブミットの確認

submission = pd.read_csv('/kaggle/input/ml-exam-20201006/sample_submission.csv', index_col=0)

# submission.to_csv('submission.csv')

# submission
import datetime

from time import time
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split as split

import category_encoders as ce

import lightgbm as lgb
#メモリ・セーブ

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics: 

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = pd.read_csv('/kaggle/input/ml-exam-20201006/train.csv')

# train = reduce_mem_usage(train)

print(train.shape)
test = pd.read_csv('/kaggle/input/ml-exam-20201006/test.csv')

# test = reduce_mem_usage(test)

print(test.shape)
city_info = pd.read_csv('/kaggle/input/ml-exam-20201006/city_info.csv')

# city_info = reduce_mem_usage(city_info)

print(city_info.shape)
station_info = pd.read_csv('/kaggle/input/ml-exam-20201006/station_info.csv')

# station_info = reduce_mem_usage(station_info)

print(station_info.shape)
train.head(5)
test.head(5)
city_info.head(5)
station_info.head(5)
def hist_train_vs_test(feature,bins,clip = False):

    plt.figure(figsize=(16, 8))

    if clip:

        th_train = np.percentile(train[feature], 99)

        th_test = np.percentile(test[feature], 99)

        plt.hist(x=[train[train[feature]<th_train][feature], test[test[feature]<th_test][feature]])

    else:

        plt.hist(x=[train[feature], test[feature]])

    plt.legend(['train', 'test'])

    plt.show()
train.columns
hist_train_vs_test('Prefecture',50,False)
hist_train_vs_test('MinTimeToNearestStation',30,False)
hist_train_vs_test('Area',50,False)
hist_train_vs_test('TotalFloorArea',50,False)
hist_train_vs_test('BuildingYear',100,False)
# hist_train_vs_test('CityPlanning',20,False)
# hist_train_vs_test('Renovation',50,False)
train['Municipality'].value_counts()
train['PrewarBuilding'].value_counts()
train['DistrictName'].value_counts()
train['NearestStation'].value_counts()
train['TimeToNearestStation'].value_counts()
train['FloorPlan'].value_counts()
train['LandShape'].value_counts()
train['Structure'].value_counts()
train['Use'].value_counts()
train['Purpose'].value_counts()
train['Direction'].value_counts()
train['Classification'].value_counts()
test['CityPlanning'].value_counts()
train['Renovation'].value_counts()
train['Remarks'].value_counts()
plt.hist(np.log(train['TradePrice']))
#新宿駅からの距離 Shinjuku	35.6896067	139.7005713 

station_info['dis_lat_from_shinjuku'] = (station_info['Latitude']- 35.6896067)**2

station_info['dis_lon_from_shinjuku'] = (station_info['Longitude']- 139.7005713)**2

station_info['dis_from_shinjuku'] = station_info['dis_lat_from_shinjuku'] + station_info['dis_lon_from_shinjuku']

# station_info['nop_par_dis_shinjuku'] = 775386 / station_info['dis_from_shinjuku']
#池袋駅からの距離 Ikebukuro	35.7295384	139.7131303

station_info['dis_lat_from_ikebukuro'] = (station_info['Latitude']- 35.7295384)**2

station_info['dis_lon_from_ikebukuro'] = (station_info['Longitude']- 139.7131303)**2

station_info['dis_from_ikebukuro'] = station_info['dis_lat_from_ikebukuro'] + station_info['dis_lon_from_ikebukuro']

# station_info['nop_par_dis_ikebukuro'] = 558623 / station_info['dis_from_ikebukuro']
#東京駅からの距離 Tokyo	35.6812362	139.7671248

station_info['dis_lat_from_tokyo'] = (station_info['Latitude']- 35.6812362)**2

station_info['dis_lon_from_tokyo'] = (station_info['Longitude']- 139.7671248)**2

station_info['dis_from_tokyo'] = station_info['dis_lat_from_tokyo'] + station_info['dis_lon_from_tokyo']

# station_info['nop_par_dis_tokyo'] = 462589 / station_info['dis_from_tokyo']
#横浜駅からの距離 Yokohama	35.4657858	139.6223132

station_info['dis_lat_from_yokohama'] = (station_info['Latitude']- 35.4657858)**2

station_info['dis_lon_from_yokohama'] = (station_info['Longitude']- 139.6223132)**2

station_info['dis_from_yokohama'] = station_info['dis_lat_from_yokohama'] + station_info['dis_lon_from_yokohama']

# station_info['nop_par_dis_yokohama'] = 419440 / station_info['dis_from_yokohama']
#品川駅からの距離　Shinagawa	35.6284713	139.7387597

station_info['dis_lat_from_shinagawa'] = (station_info['Latitude']- 35.6284713)**2

station_info['dis_lon_from_shinagawa'] = (station_info['Longitude']- 139.7387597)**2

station_info['dis_from_shinagawa'] = station_info['dis_lat_from_shinagawa'] + station_info['dis_lon_from_shinagawa']

# station_info['nop_par_dis_shinagawa'] = 377337 / station_info['dis_from_shinagawa']
#渋谷駅からの距離　Shibuya	35.6580339	139.7016358

station_info['dis_lat_from_shibuya'] = (station_info['Latitude']- 35.6580339)**2

station_info['dis_lon_from_shibuya'] = (station_info['Longitude']- 139.7016358)**2

station_info['dis_from_shibuya'] = station_info['dis_lat_from_shibuya'] + station_info['dis_lon_from_shibuya']

# station_info['nop_par_dis_shibuya'] = 366128 / station_info['dis_from_shibuya']
#新橋駅からの距離　Shinbashi	35.666379	139.7583398

station_info['dis_lat_from_shinbashi'] = (station_info['Latitude']- 35.666379)**2

station_info['dis_lon_from_shinbashi'] = (station_info['Longitude']- 139.7583398)**2

station_info['dis_from_shinbashi'] = station_info['dis_lat_from_shinbashi'] + station_info['dis_lon_from_shinbashi']

# station_info['nop_par_dis_shinbashi'] = 278334 / station_info['dis_from_shinbashi']
#大宮駅からの距離 Omiya (Saitama)	35.9064485	139.6238548

station_info['dis_lat_from_oomiya'] = (station_info['Latitude']- 35.9064485)**2

station_info['dis_lon_from_oomiya'] = (station_info['Longitude']- 139.6238548)**2

station_info['dis_from_oomiya'] = station_info['dis_lat_from_oomiya'] + station_info['dis_lon_from_oomiya']

# station_info['nop_par_dis_oomiya'] = 257344 / station_info['dis_from_oomiya']
#秋葉原駅からの距離 Akihabara	35.698383	139.7730717

station_info['dis_lat_from_akihabara'] = (station_info['Latitude']- 35.698383)**2

station_info['dis_lon_from_akihabara'] = (station_info['Longitude']- 139.7730717)**2

station_info['dis_from_akihabara'] = station_info['dis_lat_from_akihabara'] + station_info['dis_lon_from_akihabara']

# station_info['nop_par_dis_akihabara'] = 248033 / station_info['dis_from_akihabara']
#北千住駅からの距離 Kitasenju	35.7496971	139.8054403

station_info['dis_lat_from_kitasenjyu'] = (station_info['Latitude']- 35.7496971)**2

station_info['dis_lon_from_kitasenjyu'] = (station_info['Longitude']- 139.8054403)**2

station_info['dis_from_kitasenjyu'] = station_info['dis_lat_from_kitasenjyu'] + station_info['dis_lon_from_kitasenjyu']

# station_info['nop_par_dis_kitasenjyu'] = 221634 / station_info['dis_from_kitasenjyu']
#上野駅からの距離 Ueno	35.7141672	139.7774091　※埼玉県でも、東部は上野の影響大の為、追加

station_info['dis_lat_from_ueno'] = (station_info['Latitude']- 35.7141672)**2

station_info['dis_lon_from_ueno'] = (station_info['Longitude']- 139.777409)**2

station_info['dis_from_ueno'] = station_info['dis_lat_from_ueno'] + station_info['dis_lon_from_ueno']

# station_info['nop_par_dis_ueno'] = 182704 / station_info['dis_from_ueno']
station_info = station_info[['Station'

                             ,'dis_lat_from_shinjuku','dis_lon_from_shinjuku','dis_from_shinjuku'

                             ,'dis_lat_from_ikebukuro','dis_lon_from_ikebukuro','dis_from_ikebukuro'

                             ,'dis_lat_from_tokyo','dis_lon_from_tokyo','dis_from_tokyo'

                             ,'dis_lat_from_yokohama','dis_lon_from_yokohama','dis_from_yokohama'

                             ,'dis_lat_from_shinagawa','dis_lon_from_shinagawa','dis_from_shinagawa'

                             ,'dis_lat_from_shibuya','dis_lon_from_shibuya','dis_from_shibuya'

                             ,'dis_lat_from_shinbashi','dis_lon_from_shinbashi','dis_from_shinbashi'

                             ,'dis_lat_from_oomiya','dis_lon_from_oomiya','dis_from_oomiya'

                             ,'dis_lat_from_akihabara','dis_lon_from_akihabara','dis_from_akihabara'

                             ,'dis_lat_from_kitasenjyu','dis_lon_from_kitasenjyu','dis_from_kitasenjyu'

                             ,'dis_lat_from_ueno','dis_lon_from_ueno','dis_from_ueno'

                             # ,'nop_par_dis_shinjuku','nop_par_dis_ikebukuro','nop_par_dis_tokyo'

                             # ,'nop_par_dis_yokohama','nop_par_dis_shinagawa','nop_par_dis_shibuya'

                             # ,'nop_par_dis_shinbashi','nop_par_dis_oomiya','nop_par_dis_akihabara'

                             # ,'nop_par_dis_kitasenjyu','nop_par_dis_ueno'

                             ]]
station_info
train = pd.merge(train,station_info,left_on ='NearestStation',right_on ='Station' , how='left')
test = pd.merge(test,station_info,left_on ='NearestStation',right_on ='Station' , how='left')
#一番地価がたかい中央区からの距離

city_info['dis_lat_from_chuo'] = (city_info['Latitude']- 35.6706392)**2

city_info['dis_lon_from_chuo'] = (city_info['Longitude']- 139.7719892)**2

city_info['dis_from_chuo'] = city_info['dis_lat_from_chuo'] + city_info['dis_lon_from_chuo']

city_info = city_info[['Prefecture','Municipality','dis_lat_from_chuo','dis_lon_from_chuo','dis_from_chuo']]
# V10 外してみる

train = pd.merge(train,city_info,on =['Prefecture','Municipality'] , how='left')
# V10 外してみる

test = pd.merge(test,city_info,on =['Prefecture','Municipality'] , how='left')
train.head(30)
test.head(30)
y = train['TradePrice']
del train['TradePrice']
print(train.shape), print(test.shape)
df = pd.concat([train,test])
df['W_Structure'] = df['Structure'].apply(lambda x: 1 if 'W' in str(x) else 0 )

df['RC_Structure'] = df['Structure'].apply(lambda x: 1 if 'RC' in str(x) else 0 )

df['SRC_Structure'] = df['Structure'].apply(lambda x: 1 if 'SRC' in str(x) else 0 )

df['LS_Structure'] = df['Structure'].apply(lambda x: 1 if 'LS' in str(x) else 0 )

df['B_Structure'] = df['Structure'].apply(lambda x: 1 if 'B' in str(x) else 0 )
df['rooms'] = df['FloorPlan'].apply(lambda x: str(x)[0])

df['LDK_FloorPlan'] = df['FloorPlan'].apply(lambda x: 1 if 'LDK' in str(x) else 0 )

df['DK_FloorPlan'] = df['FloorPlan'].apply(lambda x: 1 if 'DK' in str(x) else 0 )

df['S_FloorPlan'] = df['FloorPlan'].apply(lambda x: 1 if '+' in str(x) else 0 )
# 時間　を　分化

df['is_about_TimeToNearestStation'] = df['TimeToNearestStation'].apply(lambda x:1 if '-' in str(x) else 0 )

df.loc[df['TimeToNearestStation']=='30-60minutes','TimeToNearestStation'] = 30

df.loc[df['TimeToNearestStation']=='1H-1H30','TimeToNearestStation'] = 60

df.loc[df['TimeToNearestStation']=='1H30-2H','TimeToNearestStation'] = 90

df.loc[df['TimeToNearestStation']=='2H-','TimeToNearestStation'] = 120
df['is_null_Remarks'] = df['Remarks'].isnull()

df['private_Remarks'] = df['Remarks'].apply(lambda x: 1 if 'private' in str(x) else 0 )

df['objects_Remarks'] = df['Remarks'].apply(lambda x: 1 if 'objects' in str(x) else 0 )

df['estate_Remarks'] = df['Remarks'].apply(lambda x: 1 if 'estate' in str(x) else 0 )
df['diff_year_from_Building_year'] = df['Year']-df['BuildingYear']
df['Ward_Municipality'] = df['Municipality'].apply(lambda x: 1 if 'Ward' in str(x) else 0 )

df['Village_Municipality'] = df['Municipality'].apply(lambda x: 1 if 'Village' in str(x) else 0)

df['County_Municipality'] = df['Municipality'].apply(lambda x: 1 if 'County' in str(x) else 0)
df['log_Breadth'] = np.log(df['Breadth']+1)

df['log_TotalFloorArea'] = np.log(df['TotalFloorArea']+1)
# 再分離

train = df.iloc[:train.shape[0]]

test = df.iloc[train.shape[0]:]
cats = ['rooms','Type', 'Region','FloorPlan','LandShape','Structure','Use','Direction','Classification','Renovation','CityPlanning','Purpose']

oe = ce.OrdinalEncoder(cols=cats,return_df = False)

train[cats] = oe.fit_transform(train[cats])

test[cats] = oe.transform(test[cats])
except_cols = ['id','Prefecture','Municipality','DistrictName','NearestStation','CityPlanning','TimeToNearestStation','Remarks','Station']

use_cols = [x for x in train.columns if x not in except_cols]
X = train[use_cols]

y = np.log(y + 1)

X_test = test[use_cols]
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
city = train['Municipality']

unique_city = city.unique()

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
train.head(10)
test.head(10)
print(train.shape),print(test.shape)
start_time = time()

lgb_models = []

scores = []



params = {

    'n_jobs': -1,

    'seed': 42,

    'boosting_type': 'gbdt',

    'objective': 'regression',

#     'num_iteration': 100,           # add

    'metric': 'rmse',

#     'eval_metric': 'cappa',

    'feature_fraction':0.998495,    # add

    'bagging_fraction': 0.872417,   # mod 0.8→, = subsample

    'learning_rate': 0.02,

    'feature_fraction': 0.9,        #   = colsample_bytree

    'max_depth': 13,                # mod 10→

    'num_leaves': 1028,             # mod   # 2^max_depth < num_leaves

    'min_gain_to_split':0.085502,   # add

    'min_child_weight':1.087712,    # add

    'lambda_l1': 1,  

    'lambda_l2': 1,

    'verbose': 100,

}



# Train and make models

#for fold, (train_ids, val_ids) in enumerate(folds.split(X,y)):

#for fold, (train_ids, val_ids) in enumerate(folds.split(X)):

#for fold, (train_ids, val_ids) in enumerate(folds.split(X,y,groups)):

fold=0

for train_gr, val_gr in folds.split(unique_city):



    print('● Fold :', fold+1,'/',NFOLDS)

    #train_set = lgb.Dataset(X.iloc[train_ids], y.iloc[train_ids],

    #                       categorical_feature=categorical_features)

    #val_set = lgb.Dataset(X.iloc[val_ids], y.iloc[val_ids],

    #                     categorical_feature=categorical_features)

    tr_gr, val_gr = unique_city[train_gr], unique_city[val_gr]

    train_ids = city.isin(tr_gr)

    val_ids = city.isin(val_gr)

    train_set = lgb.Dataset(X[train_ids], y[train_ids])

    val_set = lgb.Dataset(X[val_ids], y[val_ids])

    #train_set = lgb.Dataset(X.iloc[train_ids], y.iloc[train_ids])

    #val_set = lgb.Dataset(X.iloc[val_ids], y.iloc[val_ids])



    model = lgb.train(params=params,

                      train_set=train_set,

                      valid_sets=[train_set, val_set],

                      num_boost_round=5000,

                      early_stopping_rounds=100,    # del

                      verbose_eval=200

                     )

    if fold ==0:

        importance_df = pd.DataFrame(model.feature_importance(), index=X.columns, columns=['importance'])

    else:

        importance_df += pd.DataFrame(model.feature_importance(), index=X.columns, columns=['importance'])

    #gbc_va_pred = np.exp(model.predict(X.iloc[val_ids], num_iteration=model.best_iteration))

    #gbc_va_pred[gbc_va_pred<0] = 0

    lgb_models.append(model)

    fold +=1

print('\nTime:', time() - start_time)
preds = []

for model in lgb_models:

    pred = model.predict(X_test,num_iteration=model.best_iteration)

    pred = np.exp(pred) - 1

    pred[pred<0] = 0

    pred = pred.reshape(len(X_test),1).flatten()

    preds.append(pred)
pred_df = pd.DataFrame(preds).T

pred_df['TradePrice'] = pred_df.mean(axis= 1)
pred_df
submission = pd.concat([test['id'],pred_df['TradePrice']],axis=1)

submission.to_csv('submission_10.csv',index=False)