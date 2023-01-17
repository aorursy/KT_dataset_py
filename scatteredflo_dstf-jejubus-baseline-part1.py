import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



import missingno as msno



from glob import glob

import os, random, time, gc, warnings



from tqdm import tqdm_notebook



import lightgbm as lgbm

from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import mean_squared_error



from catboost import CatBoostRegressor

from sklearn.feature_selection import RFECV





from sklearn.cluster import KMeans



from datetime import datetime



from math import sqrt



import folium

from folium import Marker, Icon, CircleMarker



from pdpbox import pdp, info_plots



warnings.filterwarnings('ignore')



pd.set_option('max_columns', 500)

pd.set_option('max_rows', 500)



%matplotlib inline
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
train = pd.read_csv('/kaggle/input/dacon-2019-jeju-bus/train.csv')

test = pd.read_csv('/kaggle/input/dacon-2019-jeju-bus/test.csv')

df_bus = pd.read_csv('/kaggle/input/dacon-2019-jeju-bus/bus_bts.csv')

submission = pd.read_csv('/kaggle/input/dacon-2019-jeju-bus/submission_sample.csv')
train['date'] = pd.to_datetime(train['date'])

test['date'] = pd.to_datetime(test['date'])

# train/test_date를 datetime으로 바꿔줌



train.dtypes.head()

# 잘 바뀌었는지 확인
display(train.head())

display(test.head())
train.shape, test.shape
#=== train/test_id를 확인해보자 ===#



print('train_id : ', train['id'].min(),' // ', train['id'].max(),' // ', len(train['id']))

print('test_id : ', test['id'].min(),' // ', test['id'].max(),' // ', len(test['id']))

# id는 Train/Test-set 각각의 key로 사용됨. 

# 특별한 의미를 지니지 않고 테이블의 각 row를 구분짓는데만 사용되기에 모델링 시 제거해줘야하는 column
#=== train/test_date를 확인해보자 ===#



print('train_date : ', train['date'].min(),' // ', train['date'].max(),' // ', len(train['date']))

print('test_date : ', test['date'].min(),' // ', test['date'].max(),' // ', len(test['date']))

# train_date는 9/1 ~ 9/30까지 데이터이며, test_date는 10/1~10/16까지의 데이터임
print(train.dtypes.head())

train.head(3)

# train_date column이 object로 되어있음 --> datetime으로 바꿔주자 (엑셀과 똑같음)
#=== train/test_date의 분포를 확인해보자 ===#



plt.figure(figsize = (12,8))      

# Figure사이즈를 설정



train['date'].value_counts().sort_index().plot(color='b', lw=2, label='train')

test['date'].value_counts().sort_index().plot(color='r',  lw=2, label='test')

# train/tet의 date별 개수(value_counts)를 계산하고 -> index기준으로 정렬(set_index) -> 그걸 plot으로 만듦



plt.legend()

plt.xlabel("date")

plt.ylabel("# of rows")

plt.title("Distribution of date in Train/Test-set");

# 부가적인 세팅값들
train_bus_route_id_set = set(train['bus_route_id'])

test_bus_route_id_set = set(test['bus_route_id'])

intersection_bus_route_id = train_bus_route_id_set.intersection(test_bus_route_id_set)

len(intersection_bus_route_id)

# train/test의 bus_route_id를 set으로 만들고 겹치는 항목(intersection)이 얼마나 있는지 출력
only_train_bus_route = train_bus_route_id_set.difference(test_bus_route_id_set)

print(len(only_train_bus_route))

only_train_bus_route

# train에만 있는 bus_route의 개수를 구해보자

# 30개가 train에만 있음을 알 수 있다.

# --> Test에는 없는데, Train에만 data가 있는 경우는 문제가 되지 않음 (이유에 대해서는 고민해보세요!)
only_test_bus_route = test_bus_route_id_set.difference(train_bus_route_id_set)

print(len(only_test_bus_route))

only_test_bus_route

# test에만 있는 bus_route의 개수를 구해보자

# 14가 test에만 있음을 알 수 있다.

# --> Test에는 있는데 Train에는 없는 data의 경우는 예측하는데 무리가 있을 수 있음(Train가지고 학습을 하는데, Test는 없으니 예측을 못함)

# --> 해당을 보완해줄 수 있는 방법에 대해 고민 필요

# --> 예를 들면 해당 bus_route_id에는 탑승 승객수가 없어서 0일수도 있으니, train에 임의로 0으로 된 데이터를 넣어주면??
display(test[test['bus_route_id'].isin(only_test_bus_route)].head())

# Test-set에만 있는 bus_route_id를 확인



display(test[test['bus_route_id'].isin(intersection_bus_route_id)].head())

# Train과 Test-set에 동시에 있는 bus_route_id를 확인
ride_columns = [col for col in test.columns if '_ride' in col] + ['bus_route_id','date']

# ride 관련된 column 들을 ride_columns로 묶어준다. (+ bus_route_id, date)



take_off_columns = [col for col in test.columns if '_takeoff' in col] + ['bus_route_id','date']

# takeoff 관련된 column 들을 take_off_columns로 묶어준다. (+ bus_route_id, date)



display(ride_columns)

display(take_off_columns)
plt.figure(figsize=(12,5))



test[test['bus_route_id'].isin(only_test_bus_route)].groupby(['date','bus_route_id'])['8~9_ride'].sum().groupby('date').mean().plot(label='only in Test-set')

test[test['bus_route_id'].isin(intersection_bus_route_id)].groupby(['date','bus_route_id'])['8~9_ride'].sum().groupby('date').mean().plot(label='Both in Train/Test-set')

plt.legend()



# Test-set에만 있는 bus_route_id와 둘다 있는 것과 비교했을때 데이터 분포(평균)의 차이가 남
display(msno.matrix(train))

display(msno.matrix(test))
display(train.isnull().sum(), test.isnull().sum())
target_col = '18~20_ride'

train[target_col].value_counts().sort_index()

# Target Variable의 분포를 확인
sns.distplot(train[target_col])

# 히스토그램을 그려보자

# 0이 굉장히 많고, right-skewed된 형태이며, 값이 매우 큰 outlier들이 존재한다.(50~250)
sns.distplot(np.log1p(train[target_col]))

# log1p transformation을 적용해봐도 정규분포에 근사한 모양을 보이지 않는다.
display(train[train[target_col] == 0].head(300).describe())

display(train[train[target_col] != 0].head(300).describe())

# Train-set의 승차관련 칼럼들의 rowsum

display( train[train[target_col]==0][ride_columns].sum(axis=1).agg(['min','max']) )



# Train-set의 하차관련 칼럼들의 rowsum

display( train[train[target_col]==0][take_off_columns].sum(axis=1).agg(['min','max']) )



# Train-set의 승하차관련 칼럼들의 rowsum

display( train[train[target_col]==0][ride_columns + take_off_columns].sum(axis=1).agg(['min','max']) )
print('target == 0, train_ttl_sum')

display(train[train[target_col] == 0].loc[:,'6~7_ride':'11~12_takeoff'].sum(axis=1).agg(['min','max','mean']))

print('target != 0, train_ttl_sum')

display(train[train[target_col] != 0].loc[:,'6~7_ride':'11~12_takeoff'].sum(axis=1).agg(['min','max','mean']))



# Target이 0 or 1일때 모두 min값이 0 이상임 --> train-set에는 ride or takeoff가 1이상일때만 집계

# ride + takeoff 데이터가 0일 경우(버스는 다니는데, 탑승/하차 승객이 없을 경우) 데이터 누적이 안됨



# Test-set에 bus_route_id가 존재하는데, train-set에는 없는 경우가 이런 이유일거라 생각됨

# ★ train-set에 없는 bus_route_id를 임의로 만들고 모든 탑승/하차 이력을 0으로 하면 효과가 있지 않을까?
print('target == 0, train_ride_sum')

display(train[train[target_col] == 0].loc[:,'6~7_ride':'11~12_ride'].sum(axis=1).agg(['min','max','mean']))

print('target != 0, train_ride_sum')

display(train[train[target_col] != 0].loc[:,'6~7_ride':'11~12_ride'].sum(axis=1).agg(['min','max','mean']))

print('target == 0, train_takeoff_sum')

display(train[train[target_col] == 0].loc[:,'7~8_takeoff':'11~12_takeoff'].sum(axis=1).agg(['min','max','mean']))

print('target != 0, train_takeoff_sum')

display(train[train[target_col] != 0].loc[:,'7~8_takeoff':'11~12_takeoff'].sum(axis=1).agg(['min','max','mean']))



train.head()
# train[train['date'] == '2019-09-01'].groupby(['bus_route_id'])['station_code'].nunique().sort_values(ascending=False)[:100]

# # Bus_route에 여러 Station code들이 들어있음

# # 100개만 출력
#=== 같은 정류장 이름이 여러번 나오는 경우 ===#

multiple_station_name = train.groupby('station_name')['station_code'].nunique()

multiple_station_name = multiple_station_name[multiple_station_name>=7]

print(multiple_station_name)



df_sample = train[train['station_name'].isin(multiple_station_name.index)][['station_code','station_name','latitude','longitude']]

df_sample = df_sample.drop_duplicates().reset_index(drop=True)

df_sample
def generateMap(default_location=[33.35098, 126.79807], default_zoom_start=10):

    base_map = folium.Map(location=default_location, 

                          control_scale=True, 

                          zoom_start=default_zoom_start)

    

    # 여러 개의 정거장에 대해서 Icon 생성하기

    for row in df_sample.itertuples():

        station_code, station_name, latitude, longitude = row[1:]

        

        # Create Icon

        if station_name == '금악리':

            icon = Icon(color='red',icon='station')

        else:

            icon = Icon(color='blue',icon='station')

                

        # Add Marker

        Marker(location=[ latitude , longitude], 

               popup=f'station_code : {station_code} station_name : {station_name}',

               icon = icon).add_to(base_map)

        

    

    base_map.save('하나의 station_name에 여러개의 station_code.html')

    return base_map



generateMap()



# 위도/경도로 Map에 표시해서 확인해보니 같은 Station name에 여러개의 station code가 생길 수 있습니다.(상행/하행선 등)

# 그래서 우리는 Station name이 아닌 고유한 정거장(Unique-Key)를 무엇으로 정해야 할지 결정해야 합니다.

# 
train.groupby(['station_code'])['station_name'].nunique().value_counts()



# Station_code로 기준을 삼았을때 Station_name이 1:1로 매칭됨을 확인할 수 있음
display(train.groupby('station_code')['latitude'].nunique().value_counts())     # Station_code를 기준으로 설정했을 때 latitude(위도)가 1:1 매칭됨

display(train.groupby('station_code')['longitude'].nunique().value_counts())    # Station_code를 기준으로 설정했을 때 longitude(경도)가 1:1 매칭됨

display(train.groupby('station_code')['in_out'].nunique().value_counts())       # Station_code를 기준으로 설정했을 때 in_out(시내/시외버스)이 1:1 매칭됨
display(train['date'].min(),train['date'].max())

# 1달간의 데이터임을 확인할 수 있음
train_date = train[['date']]

train_date['week'] = train['date'].dt.week

train_date['weekday'] = train['date'].dt.weekday

train_date2 = pd.DataFrame(train_date.groupby(['week','weekday'])['weekday'].count())

train_date2



# 보아하니 35주차의 마지막과 40주차의 처음이 걸친것 같음, 해당은 그냥 36과 39주차에 섞어주자

# --> Training-set : 35,36,37,38  // Validation-set : 39, 40





# 다른 validation 기법들도 고려해볼 수 있을 것이다. --> HOLD OUT / GROUP FOLD BY weekofmonth

# 본 강의에서는 KFOLD만를 가지고 baseline model을 만들어 보기로 한다.
train['week'] = train['date'].dt.week

test['week'] = test['date'].dt.week

# date에서 week를 만들어서 week column에 넣어줌
sample_train = train[~((train['week'] == 39) + (train['week'] == 40))].reset_index(drop=True)

sample_test = train[(train['week'] == 39) + (train['week'] == 40)].reset_index(drop=True)



# train에서 week가 39,40을 제외한 것 --> train_df --> reset_index를 통해 순서대로 잘 정리해주자

# train에서 week가 39,40인것 --> test_df --> reset_index를 통해 순서대로 잘 정리해주자

display(sample_train.week.unique(), sample_test.week.unique())

# train_df와 valid_df를 잘 분리했음
# categorical variable인 'bus_route_id','in_out','station_code','station_name' 에 대해선 label_encoding을 적용



lbl = LabelEncoder()



# Implement Label Encoding 

cat_cols = ['bus_route_id','in_out','station_code','station_name']

for col in tqdm_notebook(cat_cols ):

    temp_df = pd.concat([sample_train[[col]], sample_test[[col]]], axis=0)     # sample_train, sample_test를 label encoding하기 위해 잠시 concat 해줌

    lbl.fit(temp_df[col])

    

    # sample_train/sample_test에 label_encoding한 값을 대입함

    sample_train[col]= lbl.transform(sample_train[col])

    sample_test[col]= lbl.transform(sample_test[col])

display(sample_train.dtypes, sample_test.dtypes)
sample_train.head()
n_splits= 5

NUM_BOOST_ROUND = 100000

SEED = 1019

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.3,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }

# 모델에 쓰일 parameter 정의하기
#=== Target column 설정 ===#

target_col = '18~20_ride'

train_label = train[target_col] 



#=== 형식을 맞춰주기 위해 test의 target column을 만들어주고 Nan으로 채워줌 ===#

test[target_col] = np.NaN    



#=== X_train, X_test으로 만들때 지워줄 column들을 선택 ===#

drop_cols = ['date','id']



sample_train_label = sample_train[target_col]

sample_test_label  = sample_test[target_col]

# y_train, y_test 생성
oof_train = np.zeros((sample_train.shape[0],))

oof_test = np.zeros((sample_test.shape[0],))

cv_list = []

# train_df 및 valid_df 예측값을 저장하기 위한 OOF 만들기 & CV를 저장할 list 정의



kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)

# Kfold 정의



for ind, (trn_ind, val_ind) in tqdm_notebook(enumerate(kfolds.split(X = sample_train, y = sample_train_label))):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = sample_train.iloc[trn_ind].drop(drop_cols, 1).drop(target_col, 1), sample_train_label[trn_ind]

    X_valid , y_valid = sample_train.iloc[val_ind].drop(drop_cols, 1).drop(target_col, 1), sample_train_label[val_ind]

    

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # sample_test/sample_test에 대한 예측

    valid_pred = model.predict(X_valid)

    test_pred  = model.predict(sample_test.drop(drop_cols,1).drop(target_col,1))

    

    # CV를 저장

    cv_list.append(sqrt(mean_squared_error(y_valid, valid_pred)))

    

    # OOF에 예측값을 저장

    oof_train[val_ind] = valid_pred

    oof_test += test_pred/n_splits

    print('='*80)

    

print(f"<sample_TRAIN> OVERALL RMSE : {sqrt(mean_squared_error(sample_train_label, oof_train))}")

print(f"<sample_test>  OVERALL RMSE : {sqrt(mean_squared_error(sample_test_label, oof_test))}")

# 실제값과 예측값의 분포 비교

fig, axes = plt.subplots( 1, 2, figsize=(20, 8), sharex=True, sharey=True)



# y=x를 그리기 위하여

x_range = np.linspace(0, 300, 1000)



# <SUBPLOT 1> : sample_train에 대한 예측/실제값 비교

axes[0].scatter( sample_train_label, oof_train )

axes[0].set_xlabel("Prediction")

axes[0].set_ylabel("Real")



# y=x 그리기

axes[0].plot(x_range, x_range, color='r')



# <SUBPLOT 2> : sample_test에 대한 예측/실제값 비교

axes[1].scatter( sample_test_label, oof_test )

axes[1].set_xlabel("Prediction")

axes[1].set_ylabel("Real")



# y=x 그리기

axes[1].plot(x_range, x_range, color='r');



# Super Title 

plt.suptitle('sample_train   //   sample_test');

# sample_train의 실제값 vs 예측값 비교

plt.figure(figsize=(12,6))



sns.distplot(oof_train, color='r' , label='Prediction')

sns.distplot(sample_train_label, color='b', label='Real')

plt.legend()

plt.title("sample-Train");





# sample_test의 실제값 vs 예측값 비교

plt.figure(figsize=(12,6))



sns.distplot(oof_test, color='r' , label='Prediction')

sns.distplot(sample_test_label, color='b', label='Real')

plt.legend()

plt.title("sample-Valid");



# sample_TRAIN에 해당하는 부분보다 sample_test에 해당하는 부분의 예측력이 떨어지는 모습을 보이긴 한다.

# 실제 값보다 예측 값이 지나치게 큰 경우들이 존재하는데, 해당 경우들이 어떤 것들인지 살펴봐야겠다.

# train_label 정의

train_label = train[target_col]
# ★ categorical 변수는 One-Hot Encoding을 하는게 바람직함 --> 메모리 이슈로 Label-Encoding을 수행 



lbl = LabelEncoder()



cat_cols = ['bus_route_id','in_out','station_code','station_name']

for col in tqdm_notebook(cat_cols):

    lbl.fit(train[col])

    train[col] = lbl.transform(train[col])
# 각 모델에 대한 oof 정의

ridge_oof_train = np.zeros((train.shape[0]))

lasso_oof_train = np.zeros((train.shape[0]))

dt_oof_train = np.zeros((train.shape[0]))

rf_oof_train = np.zeros((train.shape[0]))

lgbm_oof_train = np.zeros((train.shape[0]))



# Kfold 정의

kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)



# Fold별로 학습진행

for ind, (trn_ind, val_ind) in tqdm_notebook(enumerate(kfolds.split(X = train, y = train_label))):

    

    # Train/Valid-set을 정의하기

    X_train , y_train = train.iloc[trn_ind].drop(['id','date', target_col], 1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(['id','date', target_col], 1), train_label[val_ind]

    

    # (1) Ridge

    print("---TRAINING RIDGE---")

    ridge = Ridge(random_state = 1019)

    

    ridge.fit(X_train, y_train)

    

    ridge_valid_pred = ridge.predict(X_valid)

    ridge_oof_train[val_ind] = ridge_valid_pred

    

    # (2) Lasso

    print("---TRAINING LASSO---")

    lasso = Lasso(random_state = 1019)

    

    lasso.fit(X_train, y_train)

    

    lasso_valid_pred = lasso.predict(X_valid)

    lasso_oof_train[val_ind] = lasso_valid_pred

    

    # (3) Decision Tree

    print("---TRAINING DECISION TREE---")

    dt = DecisionTreeRegressor(random_state=231)

    

    dt.fit(X_train, y_train)

    

    dt_valid_pred = dt.predict(X_valid)

    dt_oof_train[val_ind] = dt_valid_pred

    

    

    # (4) Random Forest

    print("---TRAINING RANDOM FOREST---")

    rf = RandomForestRegressor(random_state=231, n_estimators=20)

    

    rf.fit(X_train, y_train)

    

    rf_valid_pred = rf.predict(X_valid)

    rf_oof_train[val_ind] = rf_valid_pred

    

    # (5) Light GBM

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 0)

    

    # sample_valid/sample_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    print('='*80)

    

print(f"<Ridge> OVERALL RMSE         : {sqrt(mean_squared_error(train_label, ridge_oof_train))}")

print(f"<Lasso> OVERALL RMSE         : {sqrt(mean_squared_error(train_label, lasso_oof_train))}")

print(f"<Decision-Tree> OVERALL RMSE : {sqrt(mean_squared_error(train_label, dt_oof_train))}")

print(f"<Random-Forest> OVERALL RMSE : {sqrt(mean_squared_error(train_label, rf_oof_train))}")

print(f"<Light-GBM> OVERALL RMSE     : {sqrt(mean_squared_error(train_label, lgbm_oof_train))}")

#===Ridge Regression ===#



plt.figure(figsize=(24,5))



plt.bar( train.drop(drop_cols,1).drop(target_col,1).columns,  ridge.coef_ )     # Ridge의 Coef를 barplot으로 그린다.(모델 영향도)



plt.axhline(y=0, color='r', linestyle='-')     # y=0인 horizental한 선을 그린다.



plt.xticks(rotation=45)

plt.title("Coef of Ridge Model");



# latitude와 longitude의 weight를 통해서 퇴근시간에 동쪽에 위치한 버스 정류장일수록 탑승 승객이 많으며, 북쪽에 위치한 버스 정류장일수록 탑승 승객이 적다고 추정

#=== Lasso Regression ===#



plt.figure(figsize=(24,5))



plt.bar( train.drop(drop_cols,1).drop(target_col,1).columns,  lasso.coef_ )     # lasso의 Coef를 barplot으로 그린다.



plt.axhline(y=0, color='r', linestyle='-')     # y=0인 horizental한 선을 그린다.



plt.xticks(rotation=45)

plt.title("Coef of lasso Model");



# 하차 승객 수 보다는 승차 승객 수가 "퇴근 시간 탑승 승객"에 보다 큰 영향을 미침

# 출퇴근 시간보다는 정오 즈음 승객이 얼마나 탔는지에 대한 정보가 중요할 수도?
#=== Target값과의 상관관계 ===#

train.corr()[target_col].sort_values(ascending=False)
#=== Random Forest ===#

plt.figure(figsize=(24,5))



plt.bar( train.drop(drop_cols,1).drop(target_col,1).columns,  rf.feature_importances_ )     # Random Forest Feature Impotance를 barplot으로 그린다.



plt.axhline(y=0, color='r', linestyle='-')     # y=0인 horizental한 선을 그린다.



plt.xticks(rotation=45)

plt.title("Coef of Random Forest Model");
#=== LightGBM ===#

plt.figure(figsize=(24,5))



plt.bar( train.drop(drop_cols,1).drop(target_col,1).columns,  model.feature_importance() )     # LightGBM의 Feature Impotance를 barplot으로 그린다.



plt.axhline(y=0, color='r', linestyle='-')     # y=0인 horizental한 선을 그린다.



plt.xticks(rotation=45)

plt.title("Coef of Random Forest Model");



# 공간적 정보를 갖는 bus_route_id, station_code, station_name, latitude, longitude의 중요도가 다른 모델보다 높은 것을 확인 가능

# 하차 승객수 보다는 승차 승객수의 변수 중요도가 더 높은 것으로 보임
sample = train.drop(drop_cols,1).drop(target_col,1).sample(1000)

print(sample.shape)

# 전체 데이터를 사용할 시 너무 많은 시간이 소요되어 일부 샘플만 사용(1000개만 사용)
#=== PDP Plot For 11~12_ride ===#

pdp_ = pdp.pdp_isolate(

    model= model, dataset=sample, model_features=list(sample), feature='11~12_ride'

)

fig, axes = pdp.pdp_plot(pdp_, '11~12_ride')



# 
#=== PDP Plot For 8~9_takeoff ===#

pdp_ = pdp.pdp_isolate(

    model= model, dataset=sample, model_features=list(sample), feature='8~9_takeoff'

)

fig, axes = pdp.pdp_plot(pdp_, '8~9_takeoff')



# 
#=== PDP Plot For latitude ===#

pdp_ = pdp.pdp_isolate(

    model= model, dataset=sample, model_features=list(sample), feature='latitude'

)

fig, axes = pdp.pdp_plot(pdp_, 'latitude')



# 
#=== PDP Plot For longitude ===#

pdp_ = pdp.pdp_isolate(

    model= model, dataset=sample, model_features=list(sample), feature='longitude'

)

fig, axes = pdp.pdp_plot(pdp_, 'longitude')



# 
#=== Interactive PDP Plot For latitude,longitude ===#

pdp_ = pdp.pdp_interact(

    model= model, dataset=sample, model_features=list(sample), features=['latitude','longitude']

)



fig, axes = pdp.pdp_interact_plot(pdp_interact_out=pdp_,

                                  feature_names=['latitude','longitude'],

                                  plot_type='grid',

                                  x_quantile=True,

                                  plot_pdp=False)



# 
#=== Library Import ===#

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from glob import glob

import os, random, time, gc, warnings

from tqdm import tqdm_notebook



import lightgbm as lgbm

from sklearn.linear_model import Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import mean_squared_error



from catboost import CatBoostRegressor

from sklearn.feature_selection import RFECV

from sklearn.cluster import KMeans



from datetime import datetime

from math import sqrt

import folium

from folium import Marker, Icon, CircleMarker

from pdpbox import pdp, info_plots



#=== Jupyter 환경설정 ===#

warnings.filterwarnings('ignore')   # 오류 무시

pd.set_option('max_columns', 500)   # Column 500열까지 출력

pd.set_option('max_rows', 500)      # Row 500행까지 출력

%matplotlib inline



#=== 데이터 load ===#

train = pd.read_csv('/kaggle/input/dacon-2019-jeju-bus/train.csv')

test = pd.read_csv('/kaggle/input/dacon-2019-jeju-bus/test.csv')

df_bus = pd.read_csv('/kaggle/input/dacon-2019-jeju-bus/bus_bts.csv')

submission = pd.read_csv('/kaggle/input/dacon-2019-jeju-bus/submission_sample.csv')



#=== train/test_date를 datetime으로 바꿔줌 ===#

train['date'] = pd.to_datetime(train['date'])

test['date'] = pd.to_datetime(test['date'])



#=== Target column 설정 ===#

target_col = '18~20_ride'

train_label = train[target_col] 



#=== 형식을 맞춰주기 위해 test의 target column을 만들어주고 Nan으로 채워줌 ===#

test[target_col] = np.NaN    



#=== X_train, X_test으로 만들때 지워줄 column들을 선택 ===#

drop_cols = ['date','id']



#=== Label Encoding ===#

lbl = LabelEncoder()

cat_cols = ['bus_route_id','in_out','station_code','station_name']   # Label Encoding할 Column 설정

for col in tqdm_notebook( cat_cols ):

    

    # Label-Encoding을 fitting함

    lbl.fit( train[[col]].append(test[[col]]) )

    

    # train/test label_encoding한 값을 대입함

    train[col] = lbl.transform(train[col])

    test[col] = lbl.transform(test[col])

    

display(train.shape, test.shape)
train_label = train[target_col]   # train target 값 지정



#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== Hyper Parameter 설정 ===#

NUM_BOOST_ROUND = 100000

SEED = 1019

n_splits= 5

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }





#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== fold별 학습 진행 ===#

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label))):         

    #=== Train/Valid-set 정의 ===#

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols,1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols,1), train_label[val_ind]

    

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)     # LightGBM의 경우 Dataset으로 묶어서 학습을 진행함

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND,      # 앞에서 만든 dtrain, dvalid를 넣어줌

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # sample_valid/sample_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

    lgbm_test_pred = model.predict(test.drop(drop_cols,1))

    

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(sqrt( mean_squared_error( train_label, lgbm_oof_train)))



#=== Feature Importance ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

display(df_imp)



#=== Save the Submission File ===#

submission[target_col] = lgbm_oof_test

submission.to_csv('submission(6.1.baseline).csv', index=False)
#=== Feature Importance 확인 ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

df_imp 
#=== 승하차 간격을 2시간 간격으로 설정 ===#

#=== 3시간으로 나눠볼 수도 있음(실험 필요) ===# 

dawn_ride_cols = ['6~7_ride','7~8_ride']

dawn_takoff_cols = ['6~7_takeoff','7~8_takeoff']

morning_ride_cols = ['8~9_ride','9~10_ride']

morning_takeoff_cols = ['8~9_takeoff','9~10_takeoff']

noon_ride_cols = ['10~11_ride','11~12_ride']

noon_takeoff_cols = ['10~11_takeoff','11~12_takeoff']



def modify_terms(df):   # 함수로 만들어서 진행

    #=== ride columns ===#

    df['dawn_ride'] = df[dawn_ride_cols].sum(axis=1)         # 6~7, 7~8 ride column을 행방향으로 sum --> dawn_ride

    df['morning_ride'] = df[morning_ride_cols].sum(axis=1)   # 8~9, 9~10 ride column을 행방향으로 sum --> morning_ride

    df['noon_ride'] = df[noon_ride_cols].sum(axis=1)         # 10~11, 11~12 ride column을 행방향으로 sum --> noon_ride

    

    #=== takeoff columns ===#

    df['dawn_takeoff'] = df[dawn_takoff_cols].sum(axis=1)          # 6~7, 7~8 takeoff column을 행방향으로 sum --> dawn_takeoff

    df['morning_takeoff'] = df[morning_takeoff_cols].sum(axis=1)   # 8~9, 9~10 takeoff column을 행방향으로 sum --> morning_takeoff

    df['noon_takeoff'] = df[noon_takeoff_cols].sum(axis=1)         # 10~11, 11~12 takeoff column을 행방향으로 sum --> noon_takeoff

    

    #=== drop columns ===#

    drop_cols_fe = dawn_ride_cols + morning_ride_cols + noon_ride_cols + dawn_takoff_cols + morning_takeoff_cols + noon_takeoff_cols

    df = df.drop(drop_cols_fe, 1)    # 원래 있던 Column들 다 제거

    

    return df

    



train = modify_terms(train)

test = modify_terms(test)



display(train.head(),test.head())
test.drop(drop_cols,1)
train_label = train[target_col]   # train target 값 지정



#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== Hyper Parameter 설정 ===#

NUM_BOOST_ROUND = 100000

SEED = 1019

n_splits= 5

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }





#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)     # Kfold 정의



#=== fold별 학습 진행 ===#

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label))):         

    #=== Train/Valid-set 정의 ===#

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols,1).drop(target_col,1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols,1).drop(target_col,1), train_label[val_ind]

    

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)     # LightGBM의 경우 Dataset으로 묶어서 학습을 진행함

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND,      # 앞에서 만든 dtrain, dvalid를 넣어줌

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # sample_valid/sample_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

    lgbm_test_pred = model.predict(test.drop(drop_cols,1).drop(target_col,1))

    

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(sqrt( mean_squared_error( train_label, lgbm_oof_train)))



#=== Feature Importance ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

display(df_imp)



#=== Save the Submission File ===#

submission[target_col] = lgbm_oof_test

submission.to_csv('submission(6.2.dawn_time).csv', index=False)
#=== Feature Importance 확인 ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

df_imp 
#=== 요일 정보(weekday) 추가 ===#

train['weekday'] = train['date'].dt.weekday

test['weekday']  = test['date'].dt.weekday



#=== 공휴일 정보 추가 ===#

holidays = [datetime(2019, 9 ,12), datetime(2019, 9, 13), datetime(2019, 9 ,14), datetime(2019, 10,3), datetime(2019, 10,9) ]   # 9/12 추석 첫날(목), 9/13 추석 둘째날(금), 9/14 추석 셋째날(토), 10/3 개천절, 10/9 한글날

train['is_holiday'] = train['date'].apply(lambda x: x in holidays).astype(int)

test['is_holiday'] = test['date'].apply(lambda x: x in holidays).astype(int)

display(train['is_holiday'].unique(), test['is_holiday'].unique())
train.groupby(['date','bus_route_id'])['dawn_ride'].transform('mean') 
#=== (1) bus_route_id 기준 ===#



# 탑승

train['avg_dawn_ride_bus_route_id'] = train.groupby(['date','bus_route_id'])['dawn_ride'].transform('mean') 

train['avg_morning_ride_bus_route_id'] = train.groupby(['date','bus_route_id'])['morning_ride'].transform('mean') 

train['avg_noon_ride_bus_route_id'] = train.groupby(['date','bus_route_id'])['noon_ride'].transform('mean') 



test['avg_dawn_ride_bus_route_id'] = test.groupby(['date','bus_route_id'])['dawn_ride'].transform('mean') 

test['avg_morning_ride_bus_route_id'] = test.groupby(['date','bus_route_id'])['morning_ride'].transform('mean') 

test['avg_noon_ride_bus_route_id'] = test.groupby(['date','bus_route_id'])['noon_ride'].transform('mean') 



# 하차

train['avg_dawn_takeoff_bus_route_id'] = train.groupby(['date','bus_route_id'])['dawn_takeoff'].transform('mean') 

train['avg_morning_takeoff_bus_route_id'] = train.groupby(['date','bus_route_id'])['morning_takeoff'].transform('mean') 

train['avg_noon_takeoff_bus_route_id'] = train.groupby(['date','bus_route_id'])['noon_takeoff'].transform('mean') 



test['avg_dawn_takeoff_bus_route_id'] = test.groupby(['date','bus_route_id'])['dawn_takeoff'].transform('mean') 

test['avg_morning_takeoff_bus_route_id'] = test.groupby(['date','bus_route_id'])['morning_takeoff'].transform('mean') 

test['avg_noon_takeoff_bus_route_id'] = test.groupby(['date','bus_route_id'])['noon_takeoff'].transform('mean') 



# #=== (2) bus_route_id, station_code 기준 ===#

# train['avg_dawn_ride_bus_route_id_station_code'] = train.groupby(['date','bus_route_id','station_code'])['dawn_ride'].transform('mean') 

# train['avg_morning_ride_bus_route_id_station_code'] = train.groupby(['date','bus_route_id','station_code'])['morning_ride'].transform('mean') 

# train['avg_noon_ride_bus_route_id_station_code'] = train.groupby(['date','bus_route_id','station_code'])['noon_ride'].transform('mean') 



# test['avg_dawn_ride_bus_route_id_station_code'] = test.groupby(['date','bus_route_id','station_code'])['dawn_ride'].transform('mean') 

# test['avg_morning_ride_bus_route_id_station_code'] = test.groupby(['date','bus_route_id','station_code'])['morning_ride'].transform('mean') 

# test['avg_noon_ride_bus_route_id_station_code'] = test.groupby(['date','bus_route_id','station_code'])['noon_ride'].transform('mean') 



# #=== (3) station_code 기준 ===#

# train['avg_dawn_ride_station_code'] = train.groupby(['date','station_code'])['dawn_ride'].transform('mean') 

# train['avg_morning_ride_bus_station_code'] = train.groupby(['date','station_code'])['morning_ride'].transform('mean') 

# train['avg_noon_ride_station_code'] = train.groupby(['date','station_code'])['noon_ride'].transform('mean') 



# test['avg_dawn_ride_station_code'] = test.groupby(['date','station_code'])['dawn_ride'].transform('mean') 

# test['avg_morning_ride_bus_station_code'] = test.groupby(['date','station_code'])['morning_ride'].transform('mean') 

# test['avg_noon_ride_station_code'] = test.groupby(['date','station_code'])['noon_ride'].transform('mean') 



#=== Target- Mean Encoding ===#

mean_encoding_temp = train.groupby(['station_code'])[target_col].mean()  



train['station_code_te'] = train['station_code'].map(mean_encoding_temp)

test['station_code_te'] = test['station_code'].map(mean_encoding_temp)
train_label = train[target_col]   # train target 값 지정



#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== Hyper Parameter 설정 ===#

NUM_BOOST_ROUND = 100000

SEED = 1019

n_splits= 5

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }





#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))

kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)     # Kfold 정의



#=== fold별 학습 진행 ===#

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label))):         

    #=== Train/Valid-set 정의 ===#

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols,1).drop(target_col,1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols,1).drop(target_col,1), train_label[val_ind]

    

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)     # LightGBM의 경우 Dataset으로 묶어서 학습을 진행함

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND,      # 앞에서 만든 dtrain, dvalid를 넣어줌

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # sample_valid/sample_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

    lgbm_test_pred = model.predict(test.drop(drop_cols,1).drop(target_col,1))

    

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(sqrt( mean_squared_error( train_label, lgbm_oof_train)))



#=== Feature Importance ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

display(df_imp)



#=== Save the Submission File ===#

submission[target_col] = lgbm_oof_test

submission.to_csv('submission(6.3.2 Auditional_FE_Mean_Encoding).csv', index=False)
df_weather = pd.read_csv('/kaggle/input/jeju-weather/jeju_weather_dataset', encoding='cp949')

df_weather.head()
df_weather.dtypes

# 일시가 object로 되어있음 이걸 datetime으로 바꿔주고 싶음
df_weather['일시'] = pd.to_datetime(df_weather['일시'])

df_weather.dtypes

# 잘 바뀌었음
df_weather['hour'] = df_weather['일시'].dt.hour     # 시간 정보

df_weather['date'] = df_weather['일시'].dt.date     # date 정보

df_weather.head()

# date랑 hour를 구해줌
# DACON에서 8/31 ~ 10/16 데이터만 사용하라고 함

df_weather = df_weather[(df_weather['일시']>='2019-08-31 00:00:00')&(df_weather['일시']<='2019-10-16 23:00:00')]

df_weather.head()
# DACON에서 해당 날짜의 15시까지 정보만 사용하라고 함

df_weather = df_weather[df_weather['hour']<=15].reset_index(drop=True)     

df_weather.head()
df_weather.groupby('일시')['강수량(mm)'].sum()

# 일자별 강수량의 합계를 구해줌
df_weather.groupby('일시')['강수량(mm)'].sum().sum()

# 혹시나 다 0인가 해서 더해봤더니 0은 아닌듯 싶음
df_weather_dict = df_weather.groupby('일시')['강수량(mm)'].sum().to_dict()

# df_weather_dict    # 해당 주석을 풀어서 확인해보세요(너무길어서 주석처리함)

# train / test에 날짜별로 강수량을 mapping 할때 사용하고자 변수를 만들어둠
train['present_weather'] = train['date'].map(df_weather_dict)

test['present_weather'] = test['date'].map(df_weather_dict)

# 앞에서 만든 dict를 mapping 시켜서 당일날 날씨를 넣어줌

train.head()
train['prev_date'] = train['date'] - pd.Timedelta('1 day')

test['prev_date'] = test['date'] - pd.Timedelta('1 day')

train.head()



# 앞에서는 당일날 날씨를 넣어줬는데, 전날 날씨도 넣어주면 좋을 것 같음,

# Mapping을 위해서 전날 날짜 Column을 만듦
train['prevday_precipitation'] = train['prev_date'].map(df_weather_dict)

test['prevday_precipitation'] = test['prev_date'].map(df_weather_dict)

train.head()

# Mapping! 옆으로 넘겨서 확인해보세요 --> 
print(train.shape, test.shape)



train = train.drop('prev_date',1)

test = test.drop('prev_date',1)     

# 이제 쓸모가 없어진 prev_date(전날 날짜) 칼럼은 삭제해줌



print(train.shape, test.shape)    

# Drop할땐 앞뒤로 shape를 찍어서 잘지워졌나 확인하는 작은 습관을...^^
train_label = train[target_col]   # train target 값 지정



#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== Hyper Parameter 설정 ===#

NUM_BOOST_ROUND = 100000

SEED = 1019

n_splits= 5

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }





#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))

kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)     # Kfold 정의



#=== fold별 학습 진행 ===#

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label))):         

    #=== Train/Valid-set 정의 ===#

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols,1).drop(target_col,1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols,1).drop(target_col,1), train_label[val_ind]

    

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)     # LightGBM의 경우 Dataset으로 묶어서 학습을 진행함

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND,      # 앞에서 만든 dtrain, dvalid를 넣어줌

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # sample_valid/sample_test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

    lgbm_test_pred = model.predict(test.drop(drop_cols,1).drop(target_col,1))

    

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(sqrt( mean_squared_error( train_label, lgbm_oof_train)))



#=== Feature Importance ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

display(df_imp)



#=== Save the Submission File ===#

submission[target_col] = lgbm_oof_test

submission.to_csv('submission(6.3.3 Auditional_FE_Weather).csv', index=False)
total = pd.concat([train,test],0)

total.head()

# BUS Routing을 펼쳐보면 Station_code를 순서대로 따라갈 것임

# 모든 Bus Routing을 확인해보기 위해 임의적으로 total을 만들어줌
total = total.reset_index(drop=True)     # 가끔 합치고 index를 안날리면 index가 꼬일수도 있습니다.

total
all_bus_route_ids = total['bus_route_id'].unique()    

# bus_route_id의 unique 값들을 변수로 지정해줌



all_bus_route_ids

# 근데 확인해보니 순서대로가 아니고 순서가 좀 엉켜있음
all_bus_route_ids = list(set(all_bus_route_ids))

# 그래서 set를 해주면 순서대로 잘 정리가 되는데, set 형태로는 향후 사용에 불편함이 있어서 list로 다시 만들어줌

# ★★★ 참고로 set을 사용하면 list에서 중복값을 제거할 수 있습니다.



all_bus_route_ids[:20]     # 너무 길어서 20개까지만 출력
#=== 많이 어려울 수 있습니다. 최대한 해석해보시고 안되면 나중에 같이해보겠습니다. ===#



bus_route_sequence = {}



for i in tqdm_notebook(all_bus_route_ids) :  # 앞에서 만든 all_bus_route_ids를 하나씩 넣어줌 # for문에 tqdm을 넣으면 로딩창이 생김

    

    df_bus_route = total[total['bus_route_id'] == i]              

    # Bus Route가 1인 경우만 추림

    

    sorted_station_codes = df_bus_route['station_code'].unique()     

    # Bus Route가 1일때 station_code들의 unique를 뽑아냄

    

    

    sorted_station_codes = pd.DataFrame(sorted_station_codes, columns=['station_code']).sort_values(['station_code'], ascending=True)   

    # station의 Unique들을 dataFrame으로 만들고, 오름차순으로 정렬

    

    sorted_station_codes['station_order'] = range(len(sorted_station_codes))   

    # 앞에 station_code들의 순번을 매기기위해서 station_order라는 column을 만들어주고, 오름차순 숫자를 넣어줌

    

    

    sorted_station_codes_dict = sorted_station_codes.groupby(['station_code'])['station_order'].sum().to_dict()

    # 앞에서 만든 station_code와 station_order를 order순으로 station_code를 정리하고 to_dict해줌(mapping을 위해)

    

    bus_route_sequence[i] = sorted_station_codes_dict

    # 이경우에는 bus_route 별로 station_code 별로 station_order(순번)을 넣어야 하기 때문에 3중 mapping 구조임

    # bus_route_sequence를 비어있는 dict로 만들어주고(해당 Cell 제일 첫줄), bus_route별로 dict구조를 넣어줌 (아래 출력값을 보고 유추 필요)

    
# bus_route_sequence  # ★ 주석을 해제해서 출력해보세요(본 커널에서는 너무 길어져서 주석처리했습니다.)



# 이런식으로 만들어집니다.

# 대략적으로 보면 Bus_route 별로, Bus_station_code 별로 몇번째 Bus_station인지를 정리가 된 것 같습니다.

# 이렇게 정리하는것도 어렵긴 한데, 앞으로도 이렇게 정리할 수도 있을것 같아 라이브러리화를 잘 해두면 도움이 많이 될 것 같습니다.
train[['bus_route_id','station_code']].apply(lambda x: bus_route_sequence.get(x[0]).get(x[1]), axis=1)



# 어렵습니다... 저도 코드 구조는 잘 이해가 안되지만...추측상 해석해보겠습니다.

# 앞에서 만든 bus_route_sequence dictionary 구조에서 bus_route_id를 첫번째 column에 mapping 시키고

# 그다음에 mapping된 dictionary 구조에서 station_code를 mapping 시켰을때 나오는 station_order(순서)를 출력해줍니다.
train['nth_station']= train[['bus_route_id','station_code']].apply(lambda x: bus_route_sequence.get(x[0]).get(x[1]), axis=1)

test['nth_station'] = test[['bus_route_id','station_code']].apply(lambda x: bus_route_sequence.get(x[0]).get(x[1]), axis=1)

# 위에서 출력된 값을 train/test의 'nth_station' column에 넣어줍니다....



train
train_label = train[target_col]   # train target 값 지정



#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== Hyper Parameter 설정 ===#

NUM_BOOST_ROUND = 100000

SEED = 1019

n_splits= 5

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }





#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))

kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)     # Kfold 정의



#=== fold별 학습 진행 ===#

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label))):         

    #=== Train/Valid-set 정의 ===#

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols,1).drop(target_col,1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols,1).drop(target_col,1), train_label[val_ind]

    

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)     # LightGBM의 경우 Dataset으로 묶어서 학습을 진행함

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND,      # 앞에서 만든 dtrain, dvalid를 넣어줌

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # valid/test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

    lgbm_test_pred = model.predict(test.drop(drop_cols,1).drop(target_col,1))

    

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(sqrt( mean_squared_error( train_label, lgbm_oof_train)))



#=== Feature Importance ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

display(df_imp)



#=== Save the Submission File ===#

submission[target_col] = lgbm_oof_test

submission.to_csv('submission(6.3.4.Auditional_FE_Bus_routing).csv', index=False)
total = pd.concat([train,test],0).reset_index(drop=True)

total
total_map = total[['latitude','longitude']]     # total dataframe에서 위도/경도에 대한 값만 취해줍니다.

display(total_map)



print('before total_shape : ',total_map.shape)

total = total_map.drop_duplicates()     # 중복된 값을 지워주는 코드 --> 실제로 잘 지워지는지 확인!

print('after total_shape : ',total_map.shape)

kmeans = KMeans(n_clusters= int(sqrt(len(total_map))), random_state=1019)     

kmeans.fit(total_map)

# 앞에서 정리한 total_map을 가지고 Kmeans를 통해 Clustering 학습시켜줘서, 지도에서 특정 구역(?)을 특정하는 데이터 만들어줌

# 참고자료 : https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-7-K-%ED%8F%89%EA%B7%A0-%EA%B5%B0%EC%A7%91%ED%99%94-K-means-Clustering



train['station_code_kmeans'] = kmeans.predict(train[['latitude','longitude']])

test['station_code_kmeans']  = kmeans.predict(test[['latitude','longitude']])

train.head()

# 앞에서 학습한 Kmeans를 가지고 train[['latitude','longitude']], test[['latitude','longitude']]의 예측값을 뽑아내고 그 값을 'station_code_kmeans'에 넣어줌

train_label = train[target_col]   # train target 값 지정



#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== Hyper Parameter 설정 ===#

NUM_BOOST_ROUND = 100000

SEED = 1019

n_splits= 5

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }





#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))

kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)     # Kfold 정의



#=== fold별 학습 진행 ===#

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label))):         

    #=== Train/Valid-set 정의 ===#

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols,1).drop(target_col,1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols,1).drop(target_col,1), train_label[val_ind]

    

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)     # LightGBM의 경우 Dataset으로 묶어서 학습을 진행함

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND,      # 앞에서 만든 dtrain, dvalid를 넣어줌

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # valid/test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

    lgbm_test_pred = model.predict(test.drop(drop_cols,1).drop(target_col,1))

    

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(sqrt( mean_squared_error( train_label, lgbm_oof_train)))



#=== Feature Importance ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

display(df_imp)



#=== Save the Submission File ===#

submission[target_col] = lgbm_oof_test

submission.to_csv('submission(6.3.5 Auditional_FE_Bus_station).csv', index=False)
lgbm_param = {'objective': 'rmse',

             'boosting_type': 'gbdt',

             'random_state': 1019,

             'learning_rate': 0.1,

             'subsample': 0.7,

             'tree_learner': 'serial',

             'colsample_bytree': 0.78,

#              'early_stopping_rounds': 50,

             'subsample_freq': 1,

             'reg_lambda': 7,

             'reg_alpha': 5,

             'num_leaves': 96,

             'seed': 1993}
#=== Target column 설정 ===#

target_col = '18~20_ride'

train_label = train[target_col] 



#=== 형식을 맞춰주기 위해 test의 target column을 만들어주고 Nan으로 채워줌 ===#

test[target_col] = np.NaN    



#=== X_train, X_test으로 만들때 지워줄 column들을 선택 ===#

drop_cols = ['date','id']
reg_model = lgbm.LGBMRegressor(**lgbm_param)

rfe = RFECV(estimator=reg_model, step=1, cv=KFold(n_splits=5, shuffle=False, random_state=231), scoring='neg_mean_squared_error', verbose=2)

rfe.fit(train.drop(drop_cols,1), train_label)



# RFECV가 어떤 의미를 가지고 있는지 google에 찾아보세요!
train_columns = pd.DataFrame(list(train.drop(drop_cols,1).drop(target_col,1).columns),columns = ['col'])

rfe_rank = pd.DataFrame(rfe.ranking_, columns=['imp'])

df_rank = pd.concat([train_columns,rfe_rank],1)

df_rank
use_cols = list(df_rank[df_rank['imp']==1]['col'])

use_cols    # df_rank의 imp(importance)가 1인 것만 use_cols로 치환(뒤에 해당 col만 사용)
lgbm_param = {'objective': 'rmse',

             'boosting_type': 'gbdt',

             'random_state': 1993,

             'learning_rate': 0.1,

             'subsample': 0.7,

             'tree_learner': 'serial',

             'colsample_bytree': 0.78,

             'early_stopping_rounds': 50,

             'subsample_freq': 1,

             'reg_lambda': 7,

             'reg_alpha': 5,

             'num_leaves': 96,

             'seed': 1993}
train_label = train[target_col]   # train target 값 지정



#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== Hyper Parameter 설정 ===#

NUM_BOOST_ROUND = 100000

SEED = 1019

n_splits= 5

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }





#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))

kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)     # Kfold 정의



#=== fold별 학습 진행 ===#

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label))):         

    #=== Train/Valid-set 정의 ===#

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols,1).drop(target_col,1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols,1).drop(target_col,1), train_label[val_ind]

    

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train[use_cols], y_train)     # ★ 앞에서 만든 use_cols을 사용해서 dataset을 만듦

    dvalid = lgbm.Dataset(X_valid[use_cols], y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND,      # 앞에서 만든 dtrain, dvalid를 넣어줌

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       verbose_eval= 100)

    

    # valid/test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid[use_cols])

    lgbm_test_pred = model.predict(test[use_cols])

    

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(sqrt( mean_squared_error( train_label, lgbm_oof_train)))



#=== Feature Importance ===#

df_imp = pd.DataFrame(data = {'col': model.feature_name(),

                              'imp': model.feature_importance()})

df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)

display(df_imp)



#=== Save the Submission File ===#

submission[target_col] = lgbm_oof_test

submission.to_csv('submission(6.4 Feature Selection & Extraction).csv', index=False)
# 모델에 쓰일 parameter 정의하기

n_splits= 5

NUM_BOOST_ROUND = 100000

SEED = 1993

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1993,

              'learning_rate':0.01,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.68,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }



n_rounds = 100000

cat_params = {

        'n_estimators': n_rounds,

        'learning_rate': 0.08,

        'eval_metric': 'RMSE',

        'loss_function': 'RMSE',

        'random_seed': 42,

        'metric_period': 500,

        'od_wait': 500,

        'task_type': 'GPU',

       'l2_leaf_reg' : 3,

        'depth': 8,

    }

# 형식을 맞춰주기 위해서 Test-set에 '18~20_ride' columns을 만들어줌

test[target_col] = np.NaN
train_label = train[target_col]   # train target 값 지정



#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))



#=== Hyper Parameter 설정 ===#

NUM_BOOST_ROUND = 100000

SEED = 1019

n_splits= 5

lgbm_param = {'objective':'rmse',

              'boosting_type': 'gbdt',

              'random_state':1019,

              'learning_rate':0.1,

              'subsample':0.7,

              'tree_learner': 'serial',

              'colsample_bytree':0.78,

              'early_stopping_rounds':50,

              'subsample_freq': 1,

              'reg_lambda':7,

              'reg_alpha': 5,

              'num_leaves': 96,

              'seed' : SEED

            }





#=== 각 모델에 대한 oof 정의 ===#

lgbm_oof_train = np.zeros((train.shape[0]))

lgbm_oof_test = np.zeros((test.shape[0]))

kfolds = KFold(n_splits=n_splits, random_state=1019, shuffle=True)     # Kfold 정의



#=== fold별 학습 진행 ===#

for ind, (trn_ind, val_ind) in tqdm_notebook( enumerate( kfolds.split( X = train, y = train_label))):         

    #=== Train/Valid-set 정의 ===#

    X_train , y_train = train.iloc[trn_ind].drop(drop_cols,1).drop(target_col,1), train_label[trn_ind]

    X_valid , y_valid = train.iloc[val_ind].drop(drop_cols,1).drop(target_col,1), train_label[val_ind]

    

    print("---TRAINING LIGHT GBM---")

    # dtrain/dvalid 정의

    dtrain = lgbm.Dataset(X_train, y_train)     # LightGBM의 경우 Dataset으로 묶어서 학습을 진행함

    dvalid = lgbm.Dataset(X_valid, y_valid)

    

    # model 정의&학습

    model = lgbm.train(lgbm_param , dtrain, NUM_BOOST_ROUND, 

                       valid_sets=(dtrain, dvalid), 

                       valid_names=('train','valid'), 

                       categorical_feature= ['bus_route_id','station_code', 'station_code_kmeans'],

                       verbose_eval= 100)

    # valid/test에 대한 예측

    lgbm_valid_pred = model.predict(X_valid)

    lgbm_test_pred = model.predict(test.drop(drop_cols, 1).drop(target_col,1))

        

    lgbm_oof_train[val_ind] = lgbm_valid_pred

    lgbm_oof_test += lgbm_test_pred/ n_splits

    print('='*80)

    

print(f"<Light-GBM> OVERALL RMSE     : {sqrt( mean_squared_error( train_label, lgbm_oof_train ) )}")



#=== Save the Submission File ===#

submission[target_col] = lgbm_oof_test

submission.to_csv('submission(6.5.Inferencing_&_Ensembling).csv', index=False)