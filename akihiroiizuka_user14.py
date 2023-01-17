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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns; sns.set(style='darkgrid')



from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, KFold

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from sklearn import preprocessing as pp

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, PowerTransformer

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor

from tqdm import tqdm_notebook as tqdm
df_train = pd.read_csv('../input/exam-for-students20200527/train.csv', index_col=0) 

df_test = pd.read_csv('../input/exam-for-students20200527/test.csv',index_col=0)

df_city = pd.read_csv('../input/exam-for-students20200527/city_info.csv')

df_station = pd.read_csv('../input/exam-for-students20200527/station_info.csv')



target = 'TradePrice'



df_train.shape, df_test.shape
df_train
df_city
# City:Prefecture以外を利用

df_cty = df_city[['Municipality', 'Latitude', 'Longitude']]



# Cityを連結

df_train = df_train.merge(df_cty, on = ['Municipality'], how = 'left')

df_test = df_test.merge(df_cty, on = ['Municipality'], how = 'left')



# Station:全て利用

df_sta = df_station.rename(columns = {'Station':'NearestStation', 'Latitude':'st_Latitude', 'Longitude':'st_Longitude'})



# Stationを連結

df_train = df_train.merge(df_sta, on = ['NearestStation'], how = 'left')

df_test = df_test.merge(df_sta, on = ['NearestStation'], how = 'left')
df_test.head()
df_train.shape, df_test.shape
df_train.dtypes
df_i = df_train.select_dtypes(include=('int64'))  # int型のカラムを抽出

df_f = df_train.select_dtypes(include=('float64'))  # float型のカラムを抽出

df_o = df_train.select_dtypes(include=('object'))  # object(Text)型のカラムを抽出

df_b = df_train.select_dtypes(include=('boolean'))  # boolean型のカラムを抽出



df_i_col = df_i.columns  # int型のカラム名をリスト化

df_f_col = df_f.columns  # float型のカラム名をリスト化

df_o_col = df_o.columns  # object型のカラム名をリスト化

df_b_col = df_b.columns  # boolean型のカラム名をリスト化
df_i.head()  # int型のカラムのみのテーブル
df_f.head()  # float型のカラムのみのテーブル
df_o.head()  # object型のカラムのみのテーブル
df_b.head()  # boolean型のカラムのみのテーブル
for col in df_train.columns:

    unq = df_train[col].nunique()

    msg = '{:>20}: {:,}'.format(col, unq)

    print(msg)
for col in df_train.columns:

    val_num = df_train[col].value_counts().sort_values(ascending=False)

    print('\nカラム名：', [col])

    print(val_num.head(10))
df_train.columns.shape
for col in df_train.columns:

    null = df_train[col].isnull().sum()  # NaN(欠損値)がTrueの合計数

    pct = null / len(df_train)

    if pct > 0:  # 0%超のみ表示

        print('{:>30}(NaN): {:,} / {:,} ({:.2%})'.format(col, null, len(df_train), pct))    
group = 'FloorPlan'  # ヒストグラムの対象データ

subgroup = 'Municipality'  # 複数時系列データのrelplot(羅列線グラフ)用

station = 'NearestStation'

pref = 'Prefecture'

fsize = 'TotalFloorArea'

built = 'BuildingYear'
fig, ax = plt.subplots(figsize=(20,6))

sns.countplot(data=df_train, x=group)
fig, ax = plt.subplots(figsize=(20,6))

sns.countplot(data=df_test, x=group)
fig, ax = plt.subplots(figsize=(20,6))

sns.countplot(data=df_train, x=pref)
fig, ax = plt.subplots(figsize=(20,6))

sns.countplot(data=df_test, x=pref)
fig, ax = plt.subplots(figsize=(20,6))

a = df_train.groupby(group).median().sort_values(by=group, ascending=True).reset_index()

sns.barplot(data=a, x=group, y=target)
fig, ax = plt.subplots(figsize=(20,6))

ran = df_train[(df_train[target] >= 0) & (df_train[target] <= 300000)]

sns.distplot(ran[target], bins=30, kde=False)
fig, ax = plt.subplots(figsize=(20,6))

ran = df_train[(df_train[fsize] >= 0) & (df_train[fsize] <= 300)]

sns.distplot(ran[fsize], bins=30, kde=False)
fig, ax = plt.subplots(figsize=(20,6))

ran = df_train[(df_train[built] >= 0) & (df_train[built] <= 2020)]

sns.distplot(ran[built], bins=30, kde=False)
df_train[group].value_counts()
df_test[group].value_counts()
fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(data=df_train.corr(), cmap='bwr')
X_train = df_train.drop([target], axis=1)

y_train = df_train[target]



X_test = df_test



# targetをlog1p変換

y_train = y_train.apply(np.log1p)
# 0であれば一致

X_train.shape[1] - X_test.shape[1]
# 0であれば一致

X_train.shape[0] - y_train.shape[0]
FloorPlan_map = {'3LDK':37, '4DK':45, '2LDK':27, '4LDK':47, '2DK':25, '1K':12, '3LDK+S':39, '5LDK':57, '3DK':35,

             '1LDK':17, '2DK+S':26, 'Open Floor':90, '1DK':15, '1R':11, '4LDK+S':49, '2K':22, '2LDK+S':29,

             '6DK':65, '1LDK+S':19, '5DK':55, '1R+S':13, '1LK':18, '1K+S':14, '3K':32, '7LDK':77, '4K':42,

             '3DK+S':36, '3D':33, '1DK+S':16, '6LDK':67, 'Studio Apartment':98, '6LDK+S':69, '4L+K':46,

             '5LDK+S':59, '7DK':75, '3LK':38, '5K':52, '2K+S':24, '8LDK':87, '3LDK+K':39, '3LD':37, '1L':15,

             '4DK+S':46, '2LK':28, 'Duplex':97, '7LDK+S':79, '4LDK+K':49, '3LD+S':39, '2LD+S':29, '8LDK+S':89,

             '4L':45, '2L':25, '2LDK+K':29, '2LK+S':29, '5LDK+K':59, '1LD+S':19, '2L+S':29, '3K+S':34,

             '1DK+K':19, '2LD':27, '1L+S':19, '2D':23, '4D':43}



Purpose_map = {'Other':6, 'House':1, 'Warehouse':4, 'Office':2, 'Factory':5, 'Shop':3}



Direction_map= {'Southwest':8, 'Northwest':4, 'East':5, 'No facing road':1, 'Northeast':3, 'Southeast':7, 'South':9, 'West':6, 'North':2}
def mapper(map_col, map_to):

    X_train[map_col] = X_train[map_col].map(map_to)

    X_test[map_col] = X_test[map_col].map(map_to)
mapper('FloorPlan', FloorPlan_map)

mapper('Purpose', Purpose_map)

mapper('Direction', Direction_map)
cats = ['FloorPlan', 'Purpose', 'Direction']
X_train[cats].head(15)
X_test[cats].head(15)
obj = []

for col in X_train.columns:

    if (X_train[col].dtype == 'object'):

        obj.append(col)
ordinal_encoder = OrdinalEncoder(cols=obj)

X_train = ordinal_encoder.fit_transform(X_train)

X_test = ordinal_encoder.transform(X_test)
X_train.fillna(-999, inplace = True)

X_test.fillna(-999, inplace = True)
X_train
X_train.dtypes
# 訓練データの欠損数  → 1件以上の欠損値を持つカラムを表示

for col in X_train.columns:

    null = X_train[col].isnull().sum()  # NaN(欠損値)がTrueの合計数

    pct = null / len(X_train)

    if pct > 0:

        print('{:>20}(NaN): {:,} / {:,} ({:.2%})'.format(col, null, len(X_train), pct))
# テストデータの欠損数 → 1件以上の欠損値を持つカラムを表示

for col in X_test.columns:

    null = X_test[col].isnull().sum()  # NaN(欠損値)がTrueの合計数

    pct = null / len(X_test)

    if pct > 0:

        print('{:>20}(NaN): {:,} / {:,} ({:.2%})'.format(col, null, len(X_test), pct))
n_itr = 3

n_split = 5

es_rounds = 20

scores = []

y_pred_cv_all = np.zeros(len(X_test))



for i in range(n_itr):

    kf = KFold(n_splits=n_split, random_state=i, shuffle=True)



    for j, (train_idx, test_idx) in tqdm(enumerate(kf.split(X_train, y_train))):

        X_train_fold, y_train_fold = X_train.values[train_idx], y_train.values[train_idx]

        X_val_fold, y_val_fold = X_train.values[test_idx], y_train.values[test_idx]

    

        clf = LGBMRegressor(n_estimators=9999, random_state=71, colsample_bytree=0.9,

                            learning_rate=0.05, min_child_samples=20, max_depth=-1,

                            min_child_weight=0.001, min_split_gain=0.0, num_leaves=15) 

           

        clf.fit(X_train_fold, y_train_fold, early_stopping_rounds=es_rounds, eval_metric='rmse', eval_set=[(X_val_fold, y_val_fold)])  



        y_pred = clf.predict(X_val_fold)

        score = mean_squared_error(y_val_fold, y_pred) ** 0.5

        scores.append(score)



        y_pred_cv_all += clf.predict(X_test)

        print(clf.predict(X_test))

            

        

print(np.mean(scores))

print(scores)



y_pred_cv_all /= (n_split * n_itr)



y_pred_exp = np.exp(y_pred_cv_all) - 1

y_pred_exp
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['Importance']).sort_values(by=['Importance'], ascending=False)

imp.head(30)
fig, ax = plt.subplots(figsize=(15, 8))

lgb.plot_importance(clf, max_num_features=30, ax=ax, importance_type='gain')
y_pred_exp
submission = pd.read_csv('../input/exam-for-students20200527/sample_submission.csv', index_col=0)

submission.TradePrice = y_pred_exp
submission.to_csv('submission.csv')

submission