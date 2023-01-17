import datetime

from time import time



import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)



import matplotlib.pyplot as plt

import seaborn as sns

import os



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import KFold



from sklearn.model_selection import train_test_split as split

import category_encoders as ce

import lightgbm as lgb
INPUT_DIR_PATH = '../input/cfds-test/'
os.listdir(INPUT_DIR_PATH)
#使用メモリ削減

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



#データ読み込み

def read_data():

    train = pd.read_csv(INPUT_DIR_PATH + 'train.csv')

    train = reduce_mem_usage(train)

    print('train has {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    test = pd.read_csv(INPUT_DIR_PATH + 'test.csv')

    test = reduce_mem_usage(test)

    print('test has {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    

    #city_info.csv

    city_info = pd.read_csv(INPUT_DIR_PATH + 'city_info.csv')

    city_info = reduce_mem_usage(city_info)

    print('city_info has {} rows and {} columns'.format(city_info.shape[0], city_info.shape[1]))

    

    #station_info

    station_info = pd.read_csv(INPUT_DIR_PATH + 'station_info.csv')

    station_info = reduce_mem_usage(station_info)

    print('station_info has {} rows and {} columns'.format(station_info.shape[0], station_info.shape[1]))

    

    return train, test, city_info, station_info
train, test, city_info, station_info = read_data()
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
#Prefectureは使ってはダメ

hist_train_vs_test('Prefecture',50,False)
#MinTimeToNearestStationは2つに分布が散っている

hist_train_vs_test('MinTimeToNearestStation',30,False)
#面積（外れ値あり）

hist_train_vs_test('Area',50,False)
#TotalFloorArea（外れ値あり）

hist_train_vs_test('TotalFloorArea',50,False)
#BuildingYear

hist_train_vs_test('BuildingYear',100,False)
#対数？Breadth

#hist_train_vs_test('CityPlanning',20se)
#CoverageRatio

#hist_train_vs_test('Renovation',50,False)
#Ward Village, Countyは使えそう

train['Municipality'].value_counts()

#train['PrewarBuilding'].value_counts()
#DistictNameはいったん使わない

train['DistrictName'].value_counts()
#NearestStationもだめ

train['NearestStation'].value_counts()
#30-60minutesと1H-1H30と1H30-2H,2H-が邪魔

#徒歩0分とは

train['TimeToNearestStation'].value_counts()
#間取りか・・・

train['FloorPlan'].value_counts()



#rectangularフラグ、Semiフラグ、trapezoidalフラグ

train['LandShape'].value_counts()
#分割

train['Structure'].value_counts()



#カンマで分割

train['Use'].value_counts()
#Useとほぼ同じ？

train['Purpose'].value_counts()
#カテゴリ

train['Direction'].value_counts()
#Classification

train['Classification'].value_counts()
#CityPlanningはちょっとあとで考える

test['CityPlanning'].value_counts()
#Renovationはフラグ化

train['Renovation'].value_counts()
#Remarks時間があったら

train['Remarks'].value_counts()
plt.hist(np.log(train['TradePrice']))
#かなりスコアがあがっていく。主要駅からの距離

station_info['dis_lat_from_tokyo'] = (station_info['Latitude']- 35.6875)**2

station_info['dis_lon_from_tokyo'] = (station_info['Longitude']- 139.75)**2

station_info['dis_from_tokyo'] = station_info['dis_lat_from_tokyo'] + station_info['dis_lon_from_tokyo']



#埼玉といえば池袋

station_info['dis_lat_from_ikebukuro'] = (station_info['Latitude']- 35.7295384)**2

station_info['dis_lon_from_ikebukuro'] = (station_info['Longitude']- 139.7131303)**2

station_info['dis_from_ikebukuro'] = station_info['dis_lat_from_ikebukuro'] + station_info['dis_lon_from_ikebukuro']



#新宿もつけちゃう

station_info['dis_lat_from_shinjuku'] = (station_info['Latitude']- 35.6896067)**2

station_info['dis_lon_from_shinjuku'] = (station_info['Longitude']- 139.7005713)**2

station_info['dis_from_shinjuku'] = station_info['dis_lat_from_shinjuku'] + station_info['dis_lon_from_shinjuku']



#渋谷

station_info['dis_lat_from_shibuya'] = (station_info['Latitude']- 35.6580339)**2

station_info['dis_lon_from_shibuya'] = (station_info['Longitude']- 139.7016358)**2

station_info['dis_from_shibuya'] = station_info['dis_lat_from_shibuya'] + station_info['dis_lon_from_shibuya']



#上野

station_info['dis_lat_from_ueno'] = (station_info['Latitude']- 35.7141672)**2

station_info['dis_lon_from_ueno'] = (station_info['Longitude']- 139.7774091)**2

station_info['dis_from_ueno'] = station_info['dis_lat_from_ueno'] + station_info['dis_lon_from_ueno']





station_info = station_info[['Station','dis_lat_from_tokyo','dis_lon_from_tokyo','dis_from_tokyo'

                             ,'dis_lat_from_ikebukuro','dis_lon_from_ikebukuro','dis_from_ikebukuro'

                             ,'dis_lat_from_shinjuku','dis_lon_from_shinjuku','dis_from_shinjuku'

                            ,'dis_lat_from_shibuya','dis_lon_from_shibuya','dis_from_shibuya'

                            ,'dis_lat_from_ueno','dis_lon_from_ueno','dis_from_ueno']]







train = pd.merge(train,station_info,left_on ='NearestStation',right_on ='Station' , how='left')



test = pd.merge(test,station_info,left_on ='NearestStation',right_on ='Station' , how='left')
#ググったら一番地価がたかいのは中央区らしいので、そこからの距離にする

city_info['dis_lat_from_chuo'] = (city_info['Latitude']- 35.6706392)**2

city_info['dis_lon_from_chuo'] = (city_info['Longitude']- 139.7719892)**2

city_info['dis_from_chuo'] = city_info['dis_lat_from_chuo'] + city_info['dis_lon_from_chuo']

city_info = city_info[['Prefecture','Municipality','dis_lat_from_chuo','dis_lon_from_chuo','dis_from_chuo']]

train = pd.merge(train,city_info,on =['Prefecture','Municipality'] , how='left')



test = pd.merge(test,city_info,on =['Prefecture','Municipality'] , how='left')

train
y = train['TradePrice']

del train['TradePrice']



df = pd.concat([train,test])
#W,RC,SRC,S,LS,B

df['W_Structure'] = df['Structure'].apply(lambda x: 1 if 'W' in str(x) else 0 )

df['RC_Structure'] = df['Structure'].apply(lambda x: 1 if 'RC' in str(x) else 0 )

df['SRC_Structure'] = df['Structure'].apply(lambda x: 1 if 'SRC' in str(x) else 0 )

df['LS_Structure'] = df['Structure'].apply(lambda x: 1 if 'LS' in str(x) else 0 )

df['B_Structure'] = df['Structure'].apply(lambda x: 1 if 'B' in str(x) else 0 )



#ちゃんとできなかったけどいいや

df['rooms'] = df['FloorPlan'].apply(lambda x: str(x)[0])

df['LDK_FloorPlan'] = df['FloorPlan'].apply(lambda x: 1 if 'LDK' in str(x) else 0 )

df['DK_FloorPlan'] = df['FloorPlan'].apply(lambda x: 1 if 'DK' in str(x) else 0 )

df['S_FloorPlan'] = df['FloorPlan'].apply(lambda x: 1 if '+' in str(x) else 0 )
#TimeToNearestStation処理

df['is_about_TimeToNearestStation'] = df['TimeToNearestStation'].apply(lambda x:1 if '-' in str(x) else 0 )

df.loc[df['TimeToNearestStation']=='30-60minutes','TimeToNearestStation'] = 30

df.loc[df['TimeToNearestStation']=='1H-1H30','TimeToNearestStation'] = 60

df.loc[df['TimeToNearestStation']=='1H30-2H','TimeToNearestStation'] = 90

df.loc[df['TimeToNearestStation']=='2H-','TimeToNearestStation'] = 120



#MinTimeToNearestStation,MaxTimeToNearestStationは不要
#対数化

df['log_Breadth'] = np.log(df['Breadth']+1)

df['log_TotalFloorArea'] = np.log(df['TotalFloorArea']+1)



df['diff_year_from_Building_year'] = df['Year']-df['BuildingYear']
#Remarks



df['is_null_Remarks'] = df['Remarks'].isnull()

df['private_Remarks'] = df['Remarks'].apply(lambda x: 1 if 'private' in str(x) else 0 )

df['objects_Remarks'] = df['Remarks'].apply(lambda x: 1 if 'objects' in str(x) else 0 )

df['estate_Remarks'] = df['Remarks'].apply(lambda x: 1 if 'estate' in str(x) else 0 )



#Municipalityは使えない(GroupKfold)

df['Ward_Municipality'] = df['Municipality'].apply(lambda x: 1 if 'Ward' in str(x) else 0 )

df['Village_Municipality'] = df['Municipality'].apply(lambda x: 1 if 'Village' in str(x) else 0)

df['County_Municipality'] = df['Municipality'].apply(lambda x: 1 if 'County' in str(x) else 0)
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

#folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)



folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

city = train['Municipality']

unique_city = city.unique()

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

train
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
submission = pd.concat([test['id'],pred_df['TradePrice']],axis=1)

submission.to_csv('submission.csv',index=False)
submission