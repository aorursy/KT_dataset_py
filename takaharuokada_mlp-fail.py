import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns



from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn import preprocessing as pp

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, PowerTransformer, quantile_transform



import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor



import eli5

from eli5.sklearn import PermutationImportance



from sklearn.svm import SVC

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor



import itertools

import math

import re



pd.options.display.max_columns = None
# 環境のパス取得

import os

if 'KAGGLE_URL_BASE' in os.environ:

    print('running in kaggle kernel')

    #data_dir = Path('/kaggle/input')

    

    # inputデータのディレクトリを指定(/で終わる)

    INPUT_DATA_DIR = '/kaggle/input/exam-for-students20200527/'



    # outputデータのディレクトリを指定(/で終わる)

    OUTPUT_DATA_DIR = './'

    

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))

            

else:

    print('running in other environment')

    #data_dir = Path('./')

    

    # inputデータのディレクトリを指定(/で終わる)

    INPUT_DATA_DIR = './input/'



    # outputデータのディレクトリを指定(/で終わる)

    OUTPUT_DATA_DIR = './ver_001/'



print('==========')

print('input_dir：' + INPUT_DATA_DIR)

print('output_dir：' + OUTPUT_DATA_DIR)

# inputデータのディレクトリを指定(/で終わる)

#INPUT_DATA_DIR = './input/'



# outputデータのディレクトリを指定(/で終わる)

#OUTPUT_DATA_DIR = './ver_001/'



# ターゲットカラム名の指定

TARGET = 'TradePrice'



# トレーニングデータと検定データをマージ時、検定データのtargetとして埋める値を指定(存在しない値を設定する)

TEST_TARGET_FILL = -1
# トレーニングデータ

tmp = INPUT_DATA_DIR + 'train.csv'

df_train_org = pd.read_csv(tmp, index_col=0)



# 検定データ

tmp = INPUT_DATA_DIR + 'test.csv'

df_test_org = pd.read_csv(tmp, index_col=0)



# その他のデータ

tmp = INPUT_DATA_DIR + 'station_info.csv'

df_station_org = pd.read_csv(tmp)



# その他のデータ

tmp = INPUT_DATA_DIR + 'city_info.csv'

df_city_org = pd.read_csv(tmp)



print('df_train_org.shape:{}  df_test_org.shape:{}'.format(df_train_org.shape, df_test_org.shape))
# インプットデータフレームを退避

df_train = df_train_org.copy()

df_test = df_test_org.copy()

df_station = df_station_org.copy()

df_city = df_city_org.copy()



# トレーニングデータと検定データをマージ(検定のtargetは-1で埋める)

df_test_with_targetcol = df_test_org.copy()

df_test_with_targetcol[TARGET] = TEST_TARGET_FILL

df_all = pd.concat([df_train, df_test_with_targetcol], axis=0)
df_train
df_test
df_station
df_city
df_train.describe()
df_test.describe()
df_all.describe()
df_train_corr = df_train.corr()

df_train_corr.to_csv(OUTPUT_DATA_DIR+'check_df_train_corr.csv')

#print(df_train_corr)

ax = plt.subplots(figsize=(7, 5))

ax = sns.heatmap(df_train_corr, vmax=1, vmin=-1, center=0, cmap = 'seismic')
df_test_corr = df_test.corr()

df_test_corr.to_csv(OUTPUT_DATA_DIR+'check_df_test_corr.csv')

#print(df_test_corr)

ax = plt.subplots(figsize=(7, 5))

ax = sns.heatmap(df_test_corr, vmax=1, vmin=-1, center=0, cmap = 'seismic')
cats = []

for col in df_train.columns:

    if (df_train[col].dtype == 'int64' or df_train[col].dtype == 'float64' ):

        cats.append(col)

cats.remove(TARGET)

print(cats)
cats = []

for col in df_train.columns:

    if (df_train[col].dtype == 'int64' or df_train[col].dtype == 'float64' ):

        cats.append(col)

cats.remove(TARGET)

print(cats)

# 赤がトレーニングデータ、青がテストデータ



dens = False

for col in cats:

    plt.figure(figsize=[15,7])

    df_train[col].hist(density=dens, alpha=0.5, bins=30, color = 'r', label = 'train')

    df_test[col].hist(density=dens, alpha=0.5, bins=30, color = 'b', label = 'test')

    plt.xlabel(col)

    plt.ylabel('density')

    plt.show()
check_df_train = df_train



for col in check_df_train.columns:

    print(check_df_train[col].value_counts())

    print('===============')
check_df_test = df_test



for col in check_df_test.columns:

    print(check_df_test[col].value_counts())

    print('===============')
# NULL文字確認(あるものだけ)

check_df_train = df_train

pd.DataFrame(check_df_train.isnull().sum()[check_df_train.isnull().sum()>0])
# NULL文字確認(あるものだけ)

check_df_test = df_test

pd.DataFrame(check_df_test.isnull().sum()[check_df_test.isnull().sum()>0])
df_train_check = df_train.copy()

df_test_check = df_test.copy()
def hist_train_test(col):

    plt.figure(figsize=[15,7])

    df_train_check[col].hist(density=dens, alpha=0.5, bins=30, color = 'r', label = 'train')

    df_test_check[col].hist(density=dens, alpha=0.5, bins=30, color = 'b', label = 'test')

    plt.xlabel(col)

    plt.ylabel('density')

    plt.show()
# 完全に分離しているのてグループ？

hist_train_test('Prefecture')
# これも分離しているのでグループか使わない

hist_train_test('Municipality')
col = 'TimeToNearestStation'

print(df_train_check[col].value_counts())

print('==========')

print(df_test_check[col].value_counts())
# そのままでいい？

hist_train_test('MinTimeToNearestStation')
# そのままでいい？

hist_train_test('MaxTimeToNearestStation')
# そのままでいい？

hist_train_test('Area')
# そのままでいい？

hist_train_test('Frontage')
# そのままでいい？

hist_train_test('TotalFloorArea')
# そのままでいい？

hist_train_test('BuildingYear')
# カンマで分けれそう

col = 'Use'

print(df_train_check[col].value_counts())

print('==========')

print(df_test_check[col].value_counts())
# そのままでいい？

hist_train_test('Breadth')
# そのままでいい？

hist_train_test('CoverageRatio')
# そのままでいい？

hist_train_test('FloorAreaRatio')
# そのままでいい？

hist_train_test('Year')
# 年とくっつける？

hist_train_test('Quarter')
# ２値

col = 'Renovation'

print(df_train_check[col].value_counts())

print('==========')

print(df_test_check[col].value_counts())
# Dealings を除いて文字列処理？

col = 'Remarks'

print(df_train_check[col].value_counts())

print('==========')

print(df_test_check[col].value_counts())
df_train_change = df_train_check.copy()

df_test_change = df_test_check.copy()

df_test_change[TARGET] = -1



df_change = pd.concat([df_train_change, df_test_change])
df_change['null_sum'] = df_change.isnull().sum(axis=1)
df_change['TimeToNearestStation'].value_counts()
mapping = {'30-60minutes':30, '1H-1H30':60, '1H30-2H':90, '2H-':120}

col = 'TimeToNearestStation'

df_change[col] = df_change[col].replace(mapping).astype(float)
df_change[col].value_counts()
df_change['FloorPlan'].str[1:].value_counts()
col = 'FloorPlan'

tmp_list_other = ['Open Floor', 'Studio Apartment', 'Duplex']



tmp_list = ['LDK','K','DK','R','LDK+S','DK+S','LK','K+S','R+S','LDK+K','LD','LK+S','LD+S',]



df_change.loc[df_change[col].isnull(), col] = '#'



new_col = col + '_other'

df_change[new_col] = False

for madori in tmp_list_other:

    df_change.loc[df_change[col]==madori, new_col] = True

    df_change.loc[df_change[col]==madori, col] = '#'



for madori in tmp_list:

    new_col = col + '_' + madori.replace(' ', '')

    df_change[new_col] = False

    df_change.loc[df_change[col].str[1:].str.contains(madori), new_col] = True



new_col = col + '_rooms'

df_change[new_col] = df_change[col].apply(lambda x: str(x)[0]) # 1文字目を取得 

df_change.loc[df_change[new_col]=='#', new_col] = 0

df_change[new_col] = df_change[new_col].astype(float)



del df_change[col]
df_change
#分けるよりもセットのほうが効果ある？

"""

# もっとスマートにできないか？

col = 'FloorPlan'

tmp_list = ['L','D','K','R','\+S','\+K']

tmp_list_other = ['Open Floor', 'Studio Apartment', 'Duplex']



df_change.loc[df_change[col].isnull(), col] = '#'



for madori in tmp_list_other:

    new_col = col + '_' + madori.replace(' ', '')

    df_change[new_col] = False

    df_change.loc[df_change[col]==madori, new_col] = True

    df_change.loc[df_change[col]==madori, col] = '#'



for madori in tmp_list:

    new_col = col + '_' + madori

    df_change[new_col] = False

    df_change.loc[df_change[col].str.contains(madori), new_col] = True



new_col = col + '_rooms'

df_change[new_col] = df_change[col].apply(lambda x: str(x)[0]) # 1文字目を取得 

df_change.loc[df_change[new_col]=='#', new_col] = 0

df_change[new_col] = df_change[new_col].astype(float)



del df_change[col]

"""
df_change
df_change['Structure'].value_counts()
col = 'Structure'

tmp_list = []

df_tmp = df_change[col].str.split(', ', expand=True)



for i in range(0, df_tmp.shape[1]):

    tmp_list = tmp_list + df_tmp[i].value_counts().index.tolist()

tmp_list = list(set(tmp_list))



df_change.loc[df_change[col].isnull(), col] = '#'



df_tmp = df_change[col].str.split(', ', expand=True)

for i in range(0, df_tmp.shape[1]):

    for struct in tmp_list:

        new_col = col + '_' + struct

        df_change.loc[df_tmp[i]==struct, new_col] = True

        

for struct in tmp_list:

    new_col = col + '_' + struct

    df_change.loc[df_change[new_col].isnull(), new_col] = False

    df_change[new_col] = df_change[new_col].astype(bool)

    

del df_change[col]
df_change
df_change['Use'].value_counts()
col = 'Use'

tmp_list = []

df_tmp = df_change[col].str.split(', ', expand=True)



for i in range(0, df_tmp.shape[1]):

    tmp_list = tmp_list + df_tmp[i].value_counts().index.tolist()

tmp_list = list(set(tmp_list))



df_change.loc[df_change[col].isnull(), col] = '#'



df_tmp = df_change[col].str.split(', ', expand=True)

for i in range(0, df_tmp.shape[1]):

    for struct in tmp_list:

        new_col = col + '_' + struct

        df_change.loc[df_tmp[i]==struct, new_col] = True

        

for struct in tmp_list:

    new_col = col + '_' + struct

    df_change.loc[df_change[new_col].isnull(), new_col] = False

    df_change[new_col] = df_change[new_col].astype(bool)

    

del df_change[col]
df_change
df_change['Renovation'].value_counts()
# もっとスマートにできないか？

col = 'Renovation'

tmp_list = ['Not yet','Done']



df_change.loc[df_change[col].isnull(), col] = '#'



for madori in tmp_list:

    new_col = col + '_' + madori.replace(' ', '')

    df_change[new_col] = False

    df_change.loc[df_change[col]==madori, new_col] = True

    df_change.loc[df_change[col]==madori, col] = '#'



del df_change[col]
df_change
for col in df_change.columns:

    if df_change[col].dtype == 'object':

        print(df_change[col].value_counts())

        print('==========')
col = 'Remarks'

tmp_list = []

df_tmp = df_change[col].str.split(', ', expand=True)



for i in range(0, df_tmp.shape[1]):

    tmp_list = tmp_list + df_tmp[i].value_counts().index.tolist()

tmp_list = list(set(tmp_list))



df_change.loc[df_change[col].isnull(), col] = '#'



df_tmp = df_change[col].str.split(', ', expand=True)

for i in range(0, df_tmp.shape[1]):

    j = 0

    for struct in tmp_list:

        new_col = col + '_' + str(j)

        df_change.loc[df_tmp[i]==struct, new_col] = True

        j += 1

        

j = 0

for struct in tmp_list:

    new_col = col + '_' + str(j)

    df_change.loc[df_change[new_col].isnull(), new_col] = False

    df_change[new_col] = df_change[new_col].astype(bool)

    j += 1

    

del df_change[col]
df_change
for col in df_change.columns:

    if df_change[col].dtype == 'object':

        print(df_change[col].value_counts())

        print('==========')
df_cate_encode = df_change.copy()
# カテゴリラベルを探す

cats = []

for col in df_cate_encode.columns:

    if (df_cate_encode[col].dtype == 'object'):

        cats.append(col)

        print(col, df_cate_encode[col].nunique())
col_list = ['Prefecture', 'Type','Region','LandShape','Purpose','Direction','Classification','CityPlanning',]



# OrdinalEncode

ordinal_col = col_list

ordinal_encoder = OrdinalEncoder(cols = ordinal_col)

df_cate_encode = ordinal_encoder.fit_transform(df_cate_encode)
df_cate_encode
df_num_encode = df_cate_encode.copy()
df_num_encode
# そのままでいい？

col_list = ['TimeToNearestStation', 'MinTimeToNearestStation', 'MaxTimeToNearestStation', 'Area', 'Frontage',

            'TotalFloorArea', 'Breadth', 'CoverageRatio', 'FloorAreaRatio']

for col in col_list:

    hist_train_test(col)
col_list = ['TimeToNearestStation', 'MinTimeToNearestStation', 'MaxTimeToNearestStation', 'Area', 'Frontage',

            'TotalFloorArea', 'Breadth']



# 標準化

for col in col_list:

    df_num_encode[col] = df_num_encode[col].apply(np.log1p)
df_num_encode
df_null = df_num_encode.copy()
print(df_null.isnull().sum()[df_null.isnull().sum()!=0])

print('---------------')

print(df_null.isnull().sum()[df_null.isnull().sum()!=0])
df_null = df_null.fillna(df_null.median())
print(df_null.isnull().sum()[df_null.isnull().sum()!=0])
df_null
del df_null['Municipality']

del df_null['DistrictName']

del df_null['NearestStation']
df_model = df_null.copy()

df_train_model = df_model[df_model[TARGET]!=-1]

df_test_model = df_model[df_model[TARGET]==-1]

del df_test_model[TARGET]
df_train_model.shape
df_test_model.shape
# ターゲットを対数化

df_train_model[TARGET] = df_train_model[TARGET].apply(np.log1p)
# 地域ごとにグループを分ける

col = 'Prefecture'

group = df_train_model[col]

del df_train_model[col]

del df_test_model[col]
y_train = df_train_model[TARGET]

X_train = df_train_model.drop(TARGET, axis=1)

X_test = df_test_model
%%time

from sklearn.model_selection import GroupKFold



num_split = 5

num_iter = 3

stop_round = 50

metric = 'rmse'

scores = []

y_pred_cva = np.zeros(len(X_test)) #cvaデータ収納用



scores = []



for h in range (num_iter):

    gkf = GroupKFold(n_splits=num_split)

    

    for i, (train_ix, test_ix) in enumerate(gkf.split(X_train, y_train, group)):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        ##### 回帰モデル #####

        clf = LGBMRegressor(n_estimators=9999, random_state=71, colsample_bytree=0.9,

                            learning_rate=0.05, min_child_samples=20, #max_depth=-1,

                            min_child_weight=0.001, min_split_gain=0.0, num_leaves=15) 



        

        # トレーニング(RMSEかRMSLEか要注意(既に対数化しているかどうかも要注意))

        clf.fit(X_train_, y_train_, early_stopping_rounds=stop_round, eval_metric=metric, eval_set=[(X_val, y_val)])  



        y_pred = clf.predict(X_val)

        score = mean_squared_error(y_val, y_pred)**0.5

        scores.append(score)



        y_pred_cva += clf.predict(X_test)

        print(clf.predict(X_test))

        ##### ここまで #####



        

print('スコア平均：{}'.format(np.mean(scores)))

print('スコア：{}'.format(scores))



y_pred_cva /= (num_split * num_iter)
print('スコア平均：{}'.format(np.mean(scores)))

print('スコア：{}'.format(scores))
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(by=['importance'], ascending=False)

imp.head(50)
features = X_train.columns[X_train.any()]

fscore = clf.booster_.feature_importance(importance_type='gain').tolist()

tmp_dict = {}



for (key, val) in zip(features, fscore):

    tmp_dict[key] = val



tmp_dict = sorted(tmp_dict.items(), key=lambda x: -x[1], reverse=True)

features_fin = []

fscore_fin = []



for (key, val) in tmp_dict:

    features_fin.extend([key])

    fscore_fin.extend([val])



fscore_fin = np.array(fscore_fin)

fig, ax = plt.subplots(figsize=(20, 20))

#lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')

plt.barh(features_fin, fscore_fin)
# 結果を残しておく

scores01 = scores

y_pred_cva01 = np.exp(y_pred_cva) - 1

clf01 = clf
submission = pd.read_csv(INPUT_DATA_DIR + 'sample_submission.csv', index_col=0)

submission[TARGET] = y_pred_cva01

submission.to_csv(OUTPUT_DATA_DIR + 'submission.csv')

submission
import pandas as pd

import numpy as np

import datetime

import random

import glob

import cv2

import os

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Input

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau

import matplotlib.pyplot as plt

import itertools

import datetime

%matplotlib inline



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

# 乱数シード固定

seed_everything(2020)



# 追加

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler

from category_encoders import OrdinalEncoder

from tensorflow.keras.layers import Input, concatenate

from tensorflow.keras.models import Model
# 変更記録用のメモ

RESULT_MEMO = '001'



# ベストモデルファイル名(拡張子を除く)

BEST_MODEL = 'cnn_best_model'



# モデルのフロー図名

MODEL_FLOW = 'cnn_flow'



# EARLYSTOPPINGのパラメータ関連

EARLYSTOPPING_PATIENCE = 5 # (default:5)->固定



# PREDICTのパラメータ関連

PREDICT_BATCH_SIZE = 64 # (default:32)->固定



# COMPILEのパラメータ関連

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):

           return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

COMPILE_LOSS = root_mean_squared_error # (default:mape)>固定

COMPILE_OPTIMIZER = 'adam' # (default:adam)>固定

COMPILE_METRICS = [root_mean_squared_error] # (default:['mape'])>固定



# FITのパラメータ関連

FIT_EPOCHS = 20 # (default:50) ->固定

BATCH_SIZE = 8 # (default:16) ->固定



# REDUCELRONPLATEAUのパラメータ関連

REDUCELRONPLATEAU_PATIENCE = 3 # (default:2) ->固定

REDUCELRONPLATEAU_FACTOR = 0.03953064380648749



# DENSE(MLP)のパラメータ関連

DENSE_UNITS_MLP = [256, 32] 

DENSE_ACTIVATION_MLP = 'relu'

DENSE_KERNEL_INITIALIZER_MLP = 'he_normal'

DENSE_DROPOUT_MLP = [0.2630948036563476, 0.1053557660178037]



# 現在時刻

DT_NOW = datetime.datetime.now()
def initialize_df_results():

    return pd.DataFrame(columns = [

        'model_no', 

        'predict',

        'status',

        'model_file',

        'png_file',

    ])
def create_mlp_init(inputShape):

    model = Sequential()

    

    # 全結合層

    model.add(Dense(units=DENSE_UNITS_MLP[0], input_shape = inputShape,

                    kernel_initializer=DENSE_KERNEL_INITIALIZER_MLP,activation=DENSE_ACTIVATION_MLP))

    model.add(Dropout(DENSE_DROPOUT_MLP[0]))

    

    for (layer, dropout) in zip(DENSE_UNITS_MLP[1:], DENSE_DROPOUT_MLP[1:]):

        model.add(Dense(units=layer, kernel_initializer=DENSE_KERNEL_INITIALIZER_MLP,activation=DENSE_ACTIVATION_MLP))

        model.add(Dropout(dropout))



    return model
def create_init(inputShape_mlp):

    

    # MLPモデル定義

    mlp_model = create_mlp_init(inputShape_mlp)

    mlp_input_setting = Input(shape=inputShape_mlp)

    mlp_encoded = mlp_model(mlp_input_setting)

    

    mlp_model.add(Dense(units=1))

    

    model = mlp_model

    

    model.compile(loss=COMPILE_LOSS, optimizer=COMPILE_OPTIMIZER, metrics=COMPILE_METRICS) 

    

    model.summary()

    

    return model
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
df_train_mlp = df_train_model.copy()

df_test_mlp = df_test_model.copy()
for col in df_train_mlp.columns:

    if df_train_mlp[col].dtype == 'bool':

        df_train_mlp.loc[df_train_mlp[col]==True, col] = 1

        df_train_mlp.loc[df_train_mlp[col]==False, col] = 0

        df_train_mlp[col] = df_train_mlp[col].astype(int)



for col in df_test_mlp.columns:

    if df_test_mlp[col].dtype == 'bool':

        df_test_mlp.loc[df_test_mlp[col]==True, col] = 1

        df_test_mlp.loc[df_test_mlp[col]==False, col] = 0

        df_test_mlp[col] = df_test_mlp[col].astype(int)
# kernelが死んだので上位20個で・・・

# kernelが死んだので上位15個で・・・

col_list = [

'Type',

'Area',

'TotalFloorArea',

'BuildingYear',

'TimeToNearestStation',

'CityPlanning',

'Classification',

'null_sum',

'FloorAreaRatio',

'Breadth',

'MinTimeToNearestStation',

'Region',

'Year',

'Frontage',

'AreaIsGreaterFlag',

]
y_train = df_train_mlp[TARGET]

X_train = df_train_mlp[col_list]

X_test = df_test_mlp[col_list]
"""

%%time

from sklearn.model_selection import GroupKFold



num_split = 3

num_iter = 1

stop_round = 50

scores = []

y_pred_cva = np.zeros(len(X_test)) #cvaデータ収納用



valid_pred = []

max_row = 1

j = 0

valid_pred



for h in range (num_iter):

    gkf = GroupKFold(n_splits=num_split)

    

    for i, (train_ix, test_ix) in enumerate(gkf.split(X_train, y_train, group)):

        train_x, train_y = X_train.values[train_ix], y_train.values[train_ix]

        test_x, test_y = X_train.values[test_ix], y_train.values[test_ix]

        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)



        print('########## 交差検定{}回目開始 ##########'.format(j + 1))



        # コールバック関数の設定

        model_file = 'best_{}.hdf5'.format(j)

        es = EarlyStopping(patience=EARLYSTOPPING_PATIENCE, mode='min', verbose=1) 

        #checkpoint = ModelCheckpoint(monitor='val_loss', filepath=model_file, save_best_only=True, mode='auto') 

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=REDUCELRONPLATEAU_PATIENCE, verbose=1,  mode='min')



        # モデルの作成

        print('run01')

        inputShape_mlp = (len(X_train.columns.tolist()), )

        model = create_init(inputShape_mlp)

        print('run02')



        # 訓練実行

        model.fit(train_x, train_y, validation_data=(test_x, test_y),epochs=FIT_EPOCHS

                  , batch_size=BATCH_SIZE, callbacks=[es, 

                                                      #checkpoint,

                                                      reduce_lr_loss])



        # 精度が最も良いモデルを呼び出す

        #model.load_weights(model_file)



        # 評価

        valid_pred = model.predict(test_x, batch_size=PREDICT_BATCH_SIZE).reshape((-1,1))

        mape_score = root_mean_squared_error(test_y, valid_pred)

        scores.append([mape_score, np.mean(valid_pred)])

        

        # 予測結果を取得

        valid_pred[j] = model.predict(X_test, batch_size=PREDICT_BATCH_SIZE).reshape((-1,1))

        



        #png_file = 'image_{}.png'.format(j)

        #plot_model(model, to_file=png_file)



        # DataFrameへの保存

        end_time = datetime.datetime.now()

        result_list = pd.Series([

            max_row, 

            mape_score,

            status,

            model_file,

        ], index=df_results.columns)

        df_results = df_results.append(result_list, ignore_index=True)



        max_row = max_row + 1

"""