# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load.

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
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/ecommerce-data/Train.csv')
test = pd.read_csv('/kaggle/input/ecommerce-data/Test.csv')

train.head()
sns.distplot(train['Selling_Price'])
train['Selling_Price'] = np.log1p(train['Selling_Price'])
sns.distplot(train['Selling_Price'])
sns.boxplot(train['Selling_Price'])
train.isnull().sum()
train.describe()
train.describe(include="O")
train.nunique()
train.shape
def create_date_features(df):
    df['Date']=pd.to_datetime(df['Date'], format= '%Y%m%d', errors = 'ignore')
   
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['Dayofweek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['DayOfyear'] = pd.to_datetime(df['Date']).dt.dayofyear
    df['Week'] = pd.to_datetime(df['Date']).dt.week 
    df['Quarter'] = pd.to_datetime(df['Date']).dt.quarter  
    df['Is_month_start'] = pd.to_datetime(df['Date']).dt.is_month_start 
    df['Is_month_end'] = pd.to_datetime(df['Date']).dt.is_month_end 
    df['Is_quarter_start'] = pd.to_datetime(df['Date']).dt.is_quarter_start
    df['Is_quarter_end'] = pd.to_datetime(df['Date']).dt.is_quarter_end 
    df['Is_year_start'] = pd.to_datetime(df['Date']).dt.is_year_start 
    df['Is_year_end'] = pd.to_datetime(df['Date']).dt.is_year_end
#     df['Semester'] = np.where(df['Quarter'].isin([1,2]),1,2)   
#     df['Is_weekday'] = np.where(df['Dayofweek'].isin([0,1,2,3,4]),1,0)
#     df['Days_in_month'] = pd.to_datetime(df['Date']).dt.days_in_month 
    return df
train = create_date_features(train)
test = create_date_features(test)
train
train['Unique_Item_category_per_product_brand'] = train.groupby(['Product_Brand'])['Item_Category'].transform('nunique')
train['Unique_Subcategory_1_per_product_brand']=train.groupby(['Product_Brand'])['Subcategory_1'].transform('nunique')
train['Unique_Subcategory_2_per_product_brand']=train.groupby(['Product_Brand'])['Subcategory_2'].transform('nunique')

test['Unique_Item_category_per_product_brand']=test.groupby(['Product_Brand'])['Item_Category'].transform('nunique')
test['Unique_Subcategory_1_per_product_brand']=test.groupby(['Product_Brand'])['Subcategory_1'].transform('nunique')
test['Unique_Subcategory_2_per_product_brand']=test.groupby(['Product_Brand'])['Subcategory_2'].transform('nunique')
train.head()
# calc = df.groupby(['Product_Brand'], axis=0).agg({'Product_Brand':['count']}).reset_index() 
# calc
# calc.columns = ['Product_Brand','Product_Brand Count']
# calc
# df = df.merge(calc, on=['Product_Brand'], how='left')
# df
train
total = pd.concat([train,test], axis=0)
total['Product_Brand'] = total['Product_Brand'].str.lstrip('B-').astype(int)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

cat=['Item_Category','Subcategory_1','Subcategory_2']
for items in cat:
    total[items]=le.fit_transform(total[items])
total.head()
train.shape
test.shape
train_final = total[:train.shape[0]]
test_final = total[train.shape[0]:]
test_final.drop(['Product','Selling_Price','Date'], axis=1 , inplace=True)

x = train_final.drop(['Product','Selling_Price','Date'], axis=1)
y = train_final['Selling_Price']
test_final = test_final.drop(['Product','Selling_Price'], 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=7)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
pipe = Pipeline([('scaler','passthrough'),
                    ( 'regressor', XGBRegressor())])
# param_grid = [{'regressor' : [XGBRegressor()],
#                'learning_rate' : [0.01,0.05,0.1],
#               'scaler' : ['passthrough']
#               },
              
#              { 'regressor' : [RandomForestRegressor()],
#               'scaler' : ['passthrough'],
#               'regressor_max_depth' : [2,3,4],
#               'regressor_n_estimators': [200,300],
#               'regressor_max_features' : ['auto', 'sqrt', 'log2']
#              }
             
#              ]
param_grid = {'learning_rate' : [0.01,0.05,0.1,0.2,0.25]}
fold = KFold(n_splits=15, shuffle=True, random_state=42)
# grid = GridSearchCV(XGBRegressor(), param_grid= param_grid, cv = 5)
# grid.fit(X_train, y_train)
model_2 = XGBRegressor(
 learning_rate =0.1,
 eval_metric='rmse',
    n_estimators=5000,
  
 )
#model.fit(X_train, y_train)
model_2.fit(X_train, y_train, eval_metric='rmse', 
          eval_set=[(X_test, y_test)], early_stopping_rounds=500, verbose=100)
model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, eval_metric='rmse',
             gamma=0, gpu_id=-1, importance_type='gain',
             interaction_constraints='', learning_rate=0.1, max_delta_step=0,
             max_depth=6, min_child_weight=1, 
             monotone_constraints='()', n_estimators=5000, n_jobs=0,
             num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
model.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
score=cross_val_score(X=X_train,y=y_train,estimator=model,scoring='neg_root_mean_squared_error',cv=fold)
np.mean(score)
test_final = sc.fit_transform(test_final)
y_pred1=model.predict(test_final)

y_pred1=np.expm1(y_pred1)

y_pred1
from lightgbm import LGBMRegressor
lgb_fit_params={"early_stopping_rounds":100, 
            "eval_metric" : 'rmse', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose':100
           }

lgb_params = {'boosting_type': 'gbdt',
 'objective': 'regression',
 'metric': 'rmse',
 'verbose': 0,
 'bagging_fraction': 0.8,
 'bagging_freq': 1,
 'lambda_l1': 0.01,
 'lambda_l2': 0.01,
 'learning_rate': 0.05,
 'max_bin': 255,
 'max_depth': 6,
 'min_data_in_bin': 1,
 'min_data_in_leaf': 1,
 'num_leaves': 31}

clf_lgb = LGBMRegressor(n_estimators=10000, **lgb_params, random_state=123456789, n_jobs=-1)
clf_lgb.fit(X_train, y_train, **lgb_fit_params)



int(clf_lgb.best_iteration_)
model_lgbm = LGBMRegressor(bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.01,
              lambda_l2=0.01, learning_rate=0.05, max_bin=255, max_depth=6,
              metric='rmse', min_data_in_bin=1, min_data_in_leaf=1,
              n_estimators=170)
model_lgbm.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
score=cross_val_score(X=X_train,y=y_train,estimator=model_lgbm,scoring='neg_root_mean_squared_error',cv=5)
np.mean(score)
y_pred2=model_lgbm.predict(test_final)

y_pred2=np.expm1(y_pred2)
y_pred=(0.3*y_pred1)+(y_pred2*0.7)
y_pred
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from math import sqrt 
from sklearn.metrics import mean_squared_error, mean_squared_log_error
errrf = []
y_pred_totrf = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(x):
    X_train, X_test = x.loc[train_index], x.loc[test_index]
    y_train, y_test = x.loc[train_index], x.loc[test_index]
    
    rf = RandomForestRegressor(random_state=42, n_estimators=200)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
#     print("RMSLE: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_rf))))

#     errrf.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_rf))))
    p = rf.predict(test_final)
    y_pred_totrf.append(p)
np.mean(errrf)  
# final = np.exp(np.mean(y_pred_totrf,0))