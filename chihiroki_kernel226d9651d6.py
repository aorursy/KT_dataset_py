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
#module import

import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm, notebook



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

from datetime import datetime

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold,KFold

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance

import xgboost as xgb

import seaborn as sns

from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier

import catboost

from pylab import rcParams

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb
#データ読み込み

train = pd.read_csv('../input/exam-for-students20200527/train.csv')

test = pd.read_csv('../input/exam-for-students20200527/test.csv')

city= pd.read_csv('../input/exam-for-students20200527/city_info.csv')

station=pd.read_csv('../input/exam-for-students20200527/station_info.csv')

sumb=pd.read_csv('../input/exam-for-students20200527/sample_submission.csv')
train2=train.drop(['TradePrice'],axis=1)

dftarget=train[['TradePrice']]
#データのコンカチ

train_city=pd.merge(train2, city, how='left',left_on=['Prefecture','Municipality'],right_on=['Prefecture',

                                                                                            'Municipality'])

test_city=pd.merge(test, city, how='left',left_on=['Prefecture','Municipality'],right_on=['Prefecture',

                                                                                            'Municipality'])

train_city2=train_city.rename(columns={'Latitude': 'Latitude_city', 'Longitude': 'Longitude_city'})

test_city2=test_city.rename(columns={'Latitude': 'Latitude_city', 'Longitude': 'Longitude_city'})

train_sta=pd.merge(train_city2, station, how='left',left_on=['NearestStation'],right_on=['Station']) 

test_sta=pd.merge(test_city2, station, how='left',left_on=['NearestStation'],right_on=['Station'])  

train_sta2=train_sta.rename(columns={'Latitude': 'Latitude_sta', 'Longitude': 'Longitude_sta'})

test_sta2=test_sta.rename(columns={'Latitude': 'Latitude_sta', 'Longitude': 'Longitude_sta'})

train3=train_sta2.drop(['Prefecture','Municipality','NearestStation','Station'],axis=1)

test3=test_sta2.drop(['Prefecture','Municipality','NearestStation','Station'],axis=1)
#文字列のユニーク数の確認

for col in train3.columns:

    if train3[col].dtype == 'object':

        print(col, train3[col].nunique())

        print(col, test3[col].nunique())
all=pd.concat([train3,test3])
#新しい特徴量の生成

all['year2']=2020-all['Year']

all['nan']=all.sum(axis=1)

#一応正規化

#all_sta=
#ラベルエンコーzディング

col_nums =['Type','Region','LandShape','Purpose','Direction','Renovation']

label_encoder = OrdinalEncoder()

label_all=all.copy()

for col in col_nums:

    label_all[col+'_lab'] = label_encoder.fit_transform(label_all[col].values)
label_all.head()
label_all_time=label_all.replace({'TimeToNearestStation': {'30-60minutes':45,

                                                          '1H-1H30':75,

                                                          '1H30-2H':105}})
#label_all['TimeToNearestStation'].value_counts()

label_all_time['TimeToNearestStation'].value_counts()
#カウントエンコーディング

col_nums =['DistrictName','TimeToNearestStation','FloorPlan','Structure','Use','Classification',

'CityPlanning','Remarks','Type','Region','LandShape','Purpose','Direction','Renovation']

co_all=label_all_time.copy()

for col in col_nums:

    co_summary = co_all[col].value_counts()

    co_all[col+'_co'] = co_all[col].map(co_summary)
#欠損値の確認

print(len(co_all))

print(co_all.isnull().sum())
#カラムドロップ

co_alldrop=co_all.drop(['Type','DistrictName','FloorPlan','LandShape','id','FloorPlan','Remarks',

                       'FrontageIsGreaterFlag','Region','TimeToNearestStation','Structure','Use',

                        'Purpose','Direction','Classification','CityPlanning','Renovation',

                       'Type_co','Remarks_co','Renovation_co','FloorPlan_co','Renovation_lab',

                        'Direction_lab','LandShape_lab','Region_lab','PrewarBuilding','Purpose_lab'],axis=1)
for col in co_alldrop.columns:

    if co_alldrop[col].dtype == 'object':

        print(col)
#とりあえず(-9999でうめてみる)

fill_all=co_alldrop.fillna(-99999)
#trainとtestに分ける

newtrain= fill_all[0:len(train2)]

newtest=fill_all[len(train2):]
newtrain
x_temp.columns
print(len(newtrain))

print(len(dftarget))

n=3

kari=[]

for i in range(n):

    X_train, X_test, y_train, y_test = train_test_split(newtrain, dftarget, random_state=i*10)

#model = lgb.LGBMRegressor(random_state=0)

#model.fit(X_train, y_train)

#y_pred = model.predict(newtest)

    train_data_set = lgb.Dataset(X_train, y_train)

    test_data_set = lgb.Dataset(X_test,  y_test, reference=train_data_set)



    params = {                                                                                               

    'boosting_type': 'gbdt',                                                                             

    'objective': 'regression_l2',                                                                           

    'metric': 'l2',                                                                             

    'num_leaves': 40,                                                                                    

    'learning_rate': 0.05,                                                                               

    'feature_fraction': 0.9,                                                                             

    'bagging_fraction': 0.8,                                                                             

    'bagging_freq': 5,   

    'lambda_l2': 2,

    }                                                                                                        

                                                                                                         

    gbm = lgb.train(params,                                                                                  

                train_data_set,                                                                               

                num_boost_round=200,                                                                      

                valid_sets=test_data_set,                                                                     

                early_stopping_rounds=10

                )   

    

    y_pred = gbm.predict(newtest, num_iteration=gbm.best_iteration)

    kari.append(y_pred)

 
a=pd.DataFrame(kari)

b=a.transpose()

b.columns=['A','B','C']

b['TradePrice']=(b['A']+b['B']+b['C'])/3

ypred=b['TradePrice'].values
pred=pd.DataFrame(y_pred,columns=['TradePrice'])

newsub=sumb.drop(['TradePrice'],axis=1)

newsub2=pd.concat([newsub,pred],axis=1)

newsub2
#欠損値の確認

#print(newtrain.isnull().sum())

#print(newtest.isnull().sum())
newsub2.to_csv('submission.csv', index=False)
importance = pd.DataFrame(gbm.feature_importance(importance_type='gain'), index=newtrain.columns,columns=['importance']).sort_values(['importance'], ascending=False)

display(importance)



#imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

#imp