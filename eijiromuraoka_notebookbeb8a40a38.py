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
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import quantile_transform

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from sklearn.metrics import mean_squared_error, mean_squared_log_error



from tqdm.notebook import tqdm as tqdm



from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor



import matplotlib.pyplot as plt

import seaborn as sns



import sys
path = '/kaggle/input/ml-exam-20201006/'



df_train = pd.read_csv(path + 'train.csv', index_col=0, low_memory=False)

df_test = pd.read_csv(path + 'test.csv', index_col=0, low_memory=False)
X_train = df_train

y_train = X_train['TradePrice']



X_train = X_train.drop(['TradePrice'], axis=1)



X_test = df_test
#外部データを読み込み

station = pd.read_csv(path + 'station_info.csv', index_col=0, low_memory=False)

city = pd.read_csv(path + 'city_info.csv', index_col=0, low_memory=False)
#cityテーブルをXテーブルにleft-join

X_train = pd.merge(X_train,city,on='Municipality',how = 'left')

X_test = pd.merge(X_test,city,on='Municipality',how = 'left')



X_train
#特徴量の追加

#欠損の数をレコードに加える

X_train['null_cnt'] = X_train.isnull().sum(axis=1)

X_test['null_cnt'] = X_test.isnull().sum(axis=1)
#東京駅からの距離の２乗+新宿駅からの距離の２乗を追加



X_train['distance'] = (X_train['Latitude']- 35.6812362)*(X_train['Latitude']- 35.6896067) + (X_train['Longitude']-139.7005713)*(X_train['Longitude']-139.7005713)+ (X_train['Latitude']- 35.6896067)*(X_train['Latitude']- 35.6896067) + (X_train['Longitude']-139.7005713)*(X_train['Longitude']-139.7005713)

X_test['distance'] = (X_test['Latitude']- 35.6812362)*(X_test['Latitude']- 35.6896067) + (X_test['Longitude']-139.7005713)*(X_test['Longitude']-139.7005713)+(X_test['Latitude']- 35.6896067)*(X_test['Latitude']- 35.6896067) + (X_test['Longitude']-139.7005713)*(X_test['Longitude']-139.7005713)



#TimeToNearestStation,MaxTimeToNearestStationを除去

X_train = X_train.drop(['TimeToNearestStation'], axis=1)

X_train = X_train.drop(['MaxTimeToNearestStation'], axis=1)



X_test = X_test.drop(['TimeToNearestStation'], axis=1)

X_test = X_test.drop(['MaxTimeToNearestStation'], axis=1)



#Municipality＋CityPlanningを追加

X_train['Municipality'+'CityPlanning'] = X_train['DistrictName']+X_train['CityPlanning'] 

X_test['Municipality'+'CityPlanning'] = X_test['DistrictName']+X_test['CityPlanning'] 







#TotalFloorArea/Areaを追加

X_train['TotalFloorArea'].fillna(0, axis=0, inplace=True)

X_train['TotalFloorArea_Area'] = X_train['TotalFloorArea']/X_train['Area'] 

X_test['TotalFloorArea'].fillna(0, axis=0, inplace=True)

X_test['TotalFloorArea_Area'] = X_test['TotalFloorArea']/X_train['Area'] 



#Municipality＋Typeを追加

#X_train['Municipality'+'Type'] = X_train['Municipality']+X_train['Type'] 

#X_test['Municipality'+'Type'] = X_test['Municipality']+X_test['Type'] 





X_train

#カテゴリ

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)
ce_oe = OrdinalEncoder(cols=cats,handle_unknown='ignore')



X_train[cats].fillna('nothing', axis=0, inplace=True)

X_test[cats].fillna('nothing', axis=0, inplace=True)



X_train_cats = X_train[cats]

X_test_cats = X_test[cats]

X_train_cats
X_train_cats = ce_oe.fit_transform(X_train_cats)

X_test_cats = ce_oe.fit_transform(X_test_cats)



X_train_cats
#数値の処理

X_train_num = X_train.drop(cats, axis=1).fillna(-9999)

X_test_num = X_test.drop(cats, axis=1).fillna(-9999)



X_train_num.head()
X_train = pd.concat([X_train_cats, X_train_num], axis=1)

X_test = pd.concat([X_test_cats, X_test_num], axis=1)



X_train.fillna(-9999, inplace=True)

X_test.fillna(-9999, inplace=True)



pd.set_option('display.max_columns', 100)

X_train.head()
scores = []



skf = StratifiedKFold(n_splits=5, random_state=77, shuffle=True)



y_pred_arr = []



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix] 

    

    clf = LGBMRegressor(    )

    

    y_train_ = np.log1p(y_train_)

    y_val = np.log1p(y_val)

    

    clf.fit(X_train_, y_train_,

            early_stopping_rounds=20,

            verbose=100,

            eval_metric='rmse',

            eval_set=[(X_val, y_val)]

           )

    

    y_pred_arr.append(clf.predict(X_test))

    

    

    y_pred = clf.predict(X_val)

    score = mean_squared_error(y_val, y_pred)**0.5

    scores.append(score)
y_pred = sum(y_pred_arr) / len(y_pred_arr)



submission = pd.read_csv(path +'sample_submission.csv', index_col=0)



submission.TradePrice = np.exp(y_pred) - 1

submission['TradePrice'] = submission['TradePrice'].round().astype(int)

submission.to_csv('mySubmission.csv')



submission