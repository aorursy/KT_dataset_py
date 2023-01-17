import numpy as np

import pandas as pd

import glob

from tqdm import tqdm



import time



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.svm import NuSVR, SVR

from sklearn.linear_model import Ridge, RidgeCV



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor



import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

%matplotlib inline
segment_csvs = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/train/*")

len_segment_csvs = len(segment_csvs)

len_segment_csvs
test_csvs = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/test/*")

len_test_csvs = test_csvs

len(len_test_csvs)
sample_submission = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
len(sample_submission)
sample_submission
segment_csvs[0]
train_1 = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train/2037160701.csv')
train_1
import matplotlib.pyplot as plt



def sensor_show(df):

    f, axes = plt.subplots(10, 1)

    f.set_size_inches((16, 8)) 

    f.tight_layout() 

    plt.subplots_adjust(bottom=-0.4)

    

    # Sensor#1 ~ #10

    for i in range(1,11):

        axes[i-1].plot(df[f'sensor_{i}'].values)

        axes[i-1].set_title('Sensor_'+str(i))

        axes[i-1].set_xlabel('time')
sensor_show(train_1)
# datatable installation with internet

#!pip install datatable==0.11.0 > /dev/null



# installation without internet

!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl



import datatable as dt
df = pd.read_csv(segment_csvs[0])

df_mean = pd.DataFrame(df.mean()).T

df_mean['id'] = segment_csvs[0].split('/')[-1].split('.')[0]

segment_csvs.remove(segment_csvs[0])



for csv in tqdm(segment_csvs):

    seg_name = csv.split('/')[-1].split('.')[0]

    df = dt.fread(csv).to_jay('train.jay')

    df = dt.fread('train.jay')

    df_ = pd.DataFrame(df.mean().to_pandas()) # df.mean() for datatable

    df_['id'] = csv.split('/')[-1].split('.')[0]

    df_mean = pd.concat([df_mean,df_])

    del df

    

df_mean.head(3)
df_train = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train.csv')

df_train.head(2)
df_mean['id'] = df_mean['id'].astype('int64')
df_mean = df_mean.join(df_train.set_index('segment_id'), on='id')

df_mean.head(3)
X_train = df_mean.drop(['id','time_to_eruption'],axis=1)

y_train = df_mean['time_to_eruption']

X_train = X_train.fillna(X_train.mean())

del df_mean
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_train_scaled
df = pd.read_csv(test_csvs[0])

df_mean_test = pd.DataFrame(df.mean()).T

df_mean_test['id'] = test_csvs[0].split('/')[-1].split('.')[0]

test_csvs.remove(test_csvs[0])



for csv in tqdm(test_csvs):

    df = dt.fread(csv).to_jay('test.jay')

    df = dt.fread("test.jay")

    df_ = pd.DataFrame(df.mean().to_pandas()) # df.mean() for datatable

    df_['id'] = csv.split('/')[-1].split('.')[0]

    df_mean_test = pd.concat([df_mean_test,df_])

    del df

df_mean_test.head(3)
X_test = df_mean_test.fillna(df_mean_test.mean())

X_test = X_test.drop(['id'],axis=1)

#del df_mean_test
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
X_test_scaled
train_set = pd.DataFrame()

train_set['segment_id'] = df_train.segment_id

train_set = train_set.set_index('segment_id')

train_set = pd.merge(train_set.reset_index(), df_train, on=['segment_id'], how='left').set_index('segment_id')



y_train = train_set['time_to_eruption']
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_train, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):



    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',

                    verbose=10000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

            

        if model_type == 'rcv':

            model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_absolute_error', cv=3)

            model.fit(X_train, y_train)

            print(model.alpha_)



            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_absolute_error(y_valid, y_pred_valid)

            print(f'Fold {fold_n}. MAE: {score:.4f}.')

            print('')

            

            y_pred = model.predict(X_test).reshape(-1,)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_absolute_error(y_valid, y_pred_valid)

            print(f'Fold {fold_n}. MAE: {score:.4f}.')

            print('')

            

            y_pred = model.predict(X_test).reshape(-1,)

        

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(mean_absolute_error(y_valid, y_pred_valid))



        prediction += y_pred    

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction

    

    else:

        return oof, prediction
import lightgbm as lgb
params = {'num_leaves': 54,

         'min_data_in_leaf': 79,

         'objective': 'huber',

         'max_depth': -1,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         # "feature_fraction": 0.8354507676881442,

         "bagging_freq": 3,

         "bagging_fraction": 0.8126672064208567,

         "bagging_seed": 11,

         "metric": 'mae',

         "verbosity": -1,

         'reg_alpha': 1.1302650970728192,

         'reg_lambda': 0.3603427518866501

         }

oof_lgb, prediction_lgb, feature_importance = train_model(params=params, model_type='lgb', plot_feature_importance=True)
xgb_params = {'eta': 0.03, 'max_depth': 10, 'subsample': 0.85, #'colsample_bytree': 0.8, 

          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 4}

oof_xgb, prediction_xgb = train_model(params=xgb_params, model_type='xgb')
model = NuSVR(gamma='scale', nu=0.75, C=10.0)

oof_svr, prediction_svr = train_model(params=None, model_type='sklearn', model=model)
plt.figure(figsize=(16, 8))

plt.plot(oof_lgb, color='b', label='lgb')

plt.plot(oof_xgb, color='teal', label='xgb')

plt.plot(oof_svr, color='red', label='svr')

plt.plot((oof_lgb + oof_xgb + oof_svr) / 3, color='gold', label='blend')

plt.legend();

plt.title('Predictions');
prediction_lgb[:10], prediction_xgb[:10], prediction_svr[:10]
submission = pd.DataFrame()

submission['segment_id'] = sample_submission.segment_id

submission['time_to_eruption'] = (prediction_lgb + prediction_xgb + prediction_svr) / 3

print(submission.head())
submission.to_csv('submission.csv', index=False)