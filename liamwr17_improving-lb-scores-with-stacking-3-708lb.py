# Basic Packages
import pandas as pd
import numpy as np
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# plotting packages
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# machine learning packages
import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection, preprocessing, metrics
from sklearn.preprocessing import Imputer
pd.set_option('display.max_columns',None)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
dtypes = {'feature_1':'int16',
          'feature_2':'int16',
          'feature_3':'int16'
         }

train_df = pd.read_csv("../input/train.csv",dtype=dtypes,parse_dates=['first_active_month'])
test_df = pd.read_csv("../input/test.csv",dtype=dtypes,parse_dates=['first_active_month'])

print(f"Training Set Shape:{train_df.shape}")
print(f"Test Set Shape:{test_df.shape}")
# Using smaller data types reduces the memory usage by ~50%
data_types = {'authorized_flag':'str',
                   'card_id':'str',
                   'city_id':'int16',
                   'category_1':'str',
                   'installments':'int16',
                   'category_3':'str',
                   'merchant_category_id':'int16',
                   'merchant_id':'str',
                   'purchase_amount':'float',
                   'category_2':'str',
                   'state_id':'int16',
                   'subsector_id':'int16'}

hist_df = pd.read_csv("../input/historical_transactions.csv",dtype=data_types,parse_dates=True)
new_trans_df = pd.read_csv('../input/new_merchant_transactions.csv',dtype=data_types,parse_dates=True)
last_hist_date = datetime.datetime(2018,2,28)
for df in [new_trans_df,hist_df]:
    print(f'Preprocessing DataFrame...')
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1,'N':0}).astype('bool')
    df['category_1'] = df['category_1'].map({'Y':1,'N':0}).astype('bool')
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['time_since_purchase_date'] = (last_hist_date-df['purchase_date']).dt.days
    #df.loc[:,'purchase_date'] = pd.DatetimeIndex(df['purchase_date']).astype(np.int64)*1**(-9)
    df['installments'] = df['installments'].replace(999,-1)
    null_cols = ['city_id','state_id','subsector_id','installments']
    nan_cols = ['city_id','state_id','subsector_id','installments','merchant_id','category_3','category_2']
    
    # Identify -1 values as nans
    for col in null_cols:
        df[col] = df[col].replace(-1,np.nan)
    
    # Fill categorical nan values with mode
    for column in nan_cols:
        fill = df.loc[:,column].mode().values[0]
        df[column].fillna(fill,inplace=True)
print('Encoding Date Times...')
for df in [hist_df,new_trans_df]:
    print('...')
    df['year'] = df['purchase_date'].dt.year.astype('int16')
    df['weekofyear'] = df['purchase_date'].dt.month.astype('int16')
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype('bool')
    df['hour'] = df['purchase_date'].dt.hour.astype('int16')

    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
    df['month_diff'] = (((datetime.datetime.today()-df['purchase_date']).dt.days)//30).astype('int16')
    df['month_diff'] += df['month_lag']
# Aggregating Historical Transactions DataFrame by card_id

print('Aggregating Historical Transactions...')

# Create dictionary of column names and aggregation functions to use
agg_func = {'authorized_flag' : ['mean'],
            'city_id' : ['nunique'],
            'category_1' : ['sum','mean'],
            'installments': ['sum','min','max','var','mean'],
            'category_3' : ['nunique'],
            'merchant_category_id':['nunique'],
            'merchant_id':['nunique'],
            'purchase_amount':['sum','mean','max','min','var'],
            'purchase_date':['max','min'],
            'time_since_purchase_date':['min','max','mean'],
            'category_2':['nunique'],
            'weekend':['sum','mean'],
            'month_lag':['min','max','mean','var'],
            'month_diff':['mean','var']
           }

# Aggregate columns based on dictionary passed to agg function
ghist_df = hist_df.groupby(['card_id']).agg(agg_func)

# Rename columns before joining train/test set
ghist_df.columns = ['hist_'+'_'.join(col).strip() for col in ghist_df.columns.values]
ghist_df.head()
# Aggregate Columns based on Dictionary for new_trans_df
print('Aggregating New Transactions DataFrame...')

gnew_trans_df = new_trans_df.groupby(['card_id']).agg(agg_func)

# Rename columns before joining train / test set
gnew_trans_df.columns = ['new_'+'_'.join(col).strip() for col in gnew_trans_df.columns.values]
gnew_trans_df.head()
# Merge with train and test set
print('Merging with training set...')
train = pd.merge(train_df,ghist_df,on='card_id',how='left')
train = pd.merge(train,gnew_trans_df,on='card_id',how='left')

print('Merging with testing set...')
test = pd.merge(test_df,ghist_df,on='card_id',how='left')
test = pd.merge(test,gnew_trans_df,on='card_id',how='left')
for df in [train,test]:
    df['hist_purchase_date_uptonow'] = (datetime.datetime.today() - 
                                      df['hist_purchase_date_max']).dt.days
    df['new_purchase_date_uptonow'] = (datetime.datetime.today() - 
                                      df['new_purchase_date_max']).dt.days
    
    dt_features = ['hist_purchase_date_max','hist_purchase_date_min',
               'new_purchase_date_max','new_purchase_date_min']
    
    # Models cannot use datetime features so they are encoded here as int64s
    for feature in dt_features:
        df[feature] = df[feature].astype(np.int64)*1e-9
# Final Train and Test Sets
display(train.head())
display(test.head())
# Encoding Date times for first_active_month
for df in [train,test]:
    df['first_month'] = df['first_active_month'].dt.month
    df['first_year'] = df['first_active_month'].dt.year
    df.drop(columns = ['first_active_month'],inplace=True)
# Dealing with outliers
train['outliers'] = 0
train.loc[train['target'] < -30,'outliers'] = 1
train['outliers'].value_counts()
target_col = train['target']

features = [name for name in train.columns if name not in ['target','card_id','new_authorized_flag_mean','outliers']]

target = train['target']
del train['target']
# Fill Nan Columns
filler = Imputer()
train.loc[:,features] = filler.fit_transform(train[features].values)
test.loc[:,features] = filler.transform(test[features].values)
# Set lgbm model params
param = {'num_leaves': 31,
         'min_data_in_leaf': 30,
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "min_child_samples":20,
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "random_state": 133,
         "nthread":4,
         "verbosity": -1}

# Validating on a stratified outlier data set to give more consistent cv across the folds
# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699

folds = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=15)
lgbm_oof = np.zeros(len(train))
lgbm_pred = np.zeros(len(test))

for fold_, (train_index,valid_index) in enumerate(folds.split(train,train['outliers'].values)):
    print(f"fold number: {fold_ + 1}")
    
    train_data = lgb.Dataset(train.iloc[train_index][features],label=target.iloc[train_index])
    val_data = lgb.Dataset(train.iloc[valid_index][features],label=target.iloc[valid_index])
    num_rounds = 10000
    clf = lgb.train(param,
                    train_data,
                    num_rounds,
                    valid_sets=[train_data,val_data],
                    verbose_eval=100,
                    early_stopping_rounds=200)
    lgbm_oof[valid_index] = clf.predict(train.iloc[valid_index][features],num_iteration=clf.best_iteration)
    lgbm_pred += clf.predict(test[features],num_iteration=clf.best_iteration)/folds.n_splits
np.sqrt(mean_squared_error(lgbm_oof, target))
##xgb model
xgb_params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'gamma': 0.1,
    'max_depth': 6,
    'eval_metric':'rmse',
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

folds= model_selection.KFold(n_splits=5, shuffle=True, random_state=15)
xgb_oof = np.zeros(len(train))
xgb_pred = np.zeros(len(test))

for fold_,(train_index,valid_index) in enumerate(folds.split(train.values, train['outliers'].values)):
    print("fold: {}/5".format(fold_+1))
    start = time.time()
    train_data = xgb.DMatrix(train.iloc[train_index][features],
                           label=target.iloc[train_index])
    valid_data = xgb.DMatrix(train.iloc[valid_index][features],
                           label=target.iloc[valid_index])
    
    xgb_evals = [(train_data, 'train'), (valid_data, 'valid')]
    num_rounds = 2000
    xgb_model = xgb.train(xgb_params, train_data, num_rounds, xgb_evals, early_stopping_rounds=50, verbose_eval=1000)
    xgb_oof[valid_index] = xgb_model.predict(xgb.DMatrix(train.iloc[valid_index][features]), ntree_limit=xgb_model.best_ntree_limit+50)
    xgb_pred += xgb_model.predict(xgb.DMatrix(test[features]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits
    print(f"fold n°{fold_+1}/5 completed after: {time.time()-start:.2f} seconds",'\n')
print(np.sqrt(mean_squared_error(xgb_oof, target)))
# model trainer for sklearn pipeline
def sk_trainer(model):
    folds = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=15)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for fold_, (train_index,valid_index) in enumerate(folds.split(train,train['outliers'].values)):
        print(f"fold: {fold_ + 1}/5...")
        start = time.time()
        train_x = train.iloc[train_index][features]
        train_y=target.iloc[train_index]
        val_x = train.iloc[valid_index][features]
        val_y = target.iloc[valid_index]
        model.fit(train_x,train_y)
        oof[valid_index] = model.predict(val_x)
        predictions += model.predict(test[features])/folds.n_splits
        print(f"~~~fold {fold_ +1} completed after: {time.time()-start:.2f} seconds~~~")
    return oof, predictions, model
rf_model = RandomForestRegressor(n_estimators=50,max_depth=10)
rf_oof, rf_pred, rf_model = sk_trainer(rf_model)
print('CV Score:',np.sqrt(mean_squared_error(rf_oof, target)))
averaged_oof = (rf_oof+lgbm_oof+xgb_oof)/3
averaged_pred = (rf_pred+lgbm_pred+xgb_pred)/3
print('CV Score:',np.sqrt(mean_squared_error(averaged_oof, target)))
x = pd.DataFrame()
x['lgbm'] = lgbm_oof
x['rf'] = rf_oof
x['xgb'] = xgb_oof

test_pred = pd.DataFrame()
test_pred['lgbm'] = lgbm_pred
test_pred['rf'] = rf_pred
test_pred['xgb'] = xgb_pred
def level_2_trainer(model):
    folds = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=10)
    oof_normal = np.zeros(len(train))
    predictions_normal = np.zeros(len(test))

    for fold_, (train_index,valid_index) in enumerate(folds.split(train,train['outliers'].values)):
        print(f"fold number: {fold_ + 1}...")
        start = time.time()
        train_x = x.iloc[train_index]
        train_y=target.iloc[train_index]
        val_x = x.iloc[valid_index]
        val_y = target.iloc[valid_index]
        model.fit(train_x,train_y)
        oof_normal[valid_index] = model.predict(val_x)
        predictions_normal += model.predict(test_pred)/folds.n_splits
        print(f"fold{fold_ +1} completed after {time.time()-start}seconds")
    return oof_normal, predictions_normal, model
bay_ridge = linear_model.BayesianRidge()
bay_oof,bay_pred, bay_model = level_2_trainer(bay_ridge)
print('CV Score:',np.sqrt(mean_squared_error(bay_oof, target)))
models = ['LGBM','RF','XGB']
for model,weight in zip(models,bay_model.coef_):
    print(f"{model} Model Weights:{weight:0.3f}")
pred_df = pd.DataFrame({"card_id":test["card_id"].values})
pred_df['lgbm_target'] = lgbm_pred
pred_df['rf_target'] = rf_pred
pred_df['xgb_target'] = xgb_pred
pred_df['avg_target'] = averaged_pred
pred_df['bayridge_target'] = bay_pred
pred_df.head()
sub_df = pd.DataFrame()
sub_df['card_id'] = pred_df['card_id']
sub_df['target'] = pred_df['lgbm_target']
#sub_df.to_csv("ELOsubmission.csv", index=False)
# Test Random Forest Score
sub_df['target'] = pred_df['rf_target']
#sub_df.to_csv("ELOsubmission.csv", index=False)
sub_df['target'] = pred_df['xgb_target']
#sub_df.to_csv("ELOsubmission.csv", index=False)
sub_df['target'] = pred_df['avg_target']
#sub_df.to_csv("ELOsubmission.csv", index=False)
sub_df['target'] = pred_df['bayridge_target']
sub_df.to_csv("ELOsubmission.csv", index=False)