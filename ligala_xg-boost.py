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
test = pd.read_csv('../input/itea-goal-prediction/goal_test.csv')

train = pd.read_csv('../input/itea-goal-prediction/goal_train.csv')

goal_submission = pd.read_csv('../input/itea-goal-prediction/goal_submission.csv')
train.columns
train
train.info()
train.describe()
import datetime



date_time_str = '2020-09-10'

date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')

k = date_time_obj.date()



train['birthDate'] = pd.to_datetime(train['birthDate'], errors='coerce')

train['birth'] = k

train['birth'] = pd.to_datetime(train['birth'], errors='coerce')



test['birthDate'] = pd.to_datetime(test['birthDate'], errors='coerce')

test['birth'] = k

test['birth'] = pd.to_datetime(test['birth'], errors='coerce')
%matplotlib inline



train.plot.scatter(x = 'x_1', y = 'y_1', figsize = (15,8))
train['role'].hist()
train['body_part'].hist()
train['matchPeriod'].hist()
train.isnull().sum()
train.isna().sum()
train['matchPeriod'] = np.where(train['matchPeriod'] == "1H", 1, 2)

train.drop(columns = 'middleName', inplace = True)



train['flang'] = np.where (((train['y_1'] > 50) & (train['foot'] == "left")), 1, 

                           np.where((train['y_1'] <= 50) & (train['foot'] == "right"), 1, 0))



train['body_part'] = np.where (train['body_part'] == "right",1, 

                               np.where(train['body_part'] == "left",-1, 

                                        np.where(train['body_part'] == "head/body", 0, 0)))

train['foot'] = np.where (train['foot'] == "right",1, 

                               np.where(train['foot'] == "left",-1, 

                                        np.where(train['foot'] == "both", 0, 1)))

train['weight'] = train['weight'].fillna(train['weight'].mean())

train['birthDate'] = ((train['birth'] - train['birthDate']).dt.days / 365.25)

train['birthDate'] = train['birthDate'].fillna(train['birthDate'].mean())



train['role'] = np.where (train['role'] == "Goalkeeper",1, 

                               np.where(train['role'] == "Defender", 1, 

                                        np.where(train['role'] == "Midfielder", 2, 3)))



train['distance'] = ((100 - train['x_1'])**2 + (50 - train['y_1'])**2)**0.5



import math 

train['ugol'] = np.where(train['y_1'] > 50, 

                         (180/math.pi*np.arcsin((((train['x_1'] - 100)**2)**0.5) / (((train['x_1'] - 100)**2 + (train['y_1']-50)**2)**0.5))).astype(float), 

                         np.where(train['y_1'] < 50, (180 - 90 - (180/math.pi*np.arcsin((((train['x_1'] - 100)**2)**0.5) / (((train['x_1'] - 100)**2 + (train['y_1']-50)**2)**0.5))).astype(float)), 0))



train['Legioner'] = np.where((train['league'] == "IT") & (train['passportArea'] == "Italy"), 0,

                            np.where((train['league'] == "SP") & (train['passportArea'] == "Spain"), 0,

                                    np.where((train['league'] == "GE") & (train['passportArea'] == "Germany"), 0,

                                            np.where((train['league'] == "FR") & (train['passportArea'] == "France"), 0,

                                                    np.where((train['league'] == "EN") & (train['passportArea'] == "England"), 0, 1)))))
test['matchPeriod'] = np.where(test['matchPeriod'] == "1H", 1, 2)

test.drop(columns = 'middleName', inplace = True)



test['flang'] = np.where (((test['y_1'] > 50) & (test['foot'] == "left")), 1, 

                           np.where((test['y_1'] <= 50) & (test['foot'] == "right"), 1, 0))



test['body_part'] = np.where (test['body_part'] == "right",1, 

                               np.where(test['body_part'] == "left",-1, 

                                        np.where(test['body_part'] == "head/body", 0, 0)))

test['foot'] = np.where (test['foot'] == "right",1, 

                               np.where(test['foot'] == "left",-1, 

                                        np.where(test['foot'] == "both", 0, 1)))

test['weight'] = test['weight'].fillna(test['weight'].mean())

test['birthDate'] = ((test['birth'] - test['birthDate']).dt.days / 365.25)

test['birthDate'] = test['birthDate'].fillna(test['birthDate'].mean())



test['role'] = np.where (test['role'] == "Goalkeeper",1, 

                               np.where(test['role'] == "Defender", 1, 

                                        np.where(test['role'] == "Midfielder", 2, 3)))



test['distance'] = ((100 - test['x_1'])**2 + (50 - test['y_1'])**2)**0.5



import math 

test['ugol'] = np.where(test['y_1'] > 50, 

                         (180/math.pi*np.arcsin((((test['x_1'] - 100)**2)**0.5) / (((test['x_1'] - 100)**2 + (test['y_1']-50)**2)**0.5))).astype(float), 

                         np.where(test['y_1'] < 50, (180 - 90 - (180/math.pi*np.arcsin((((test['x_1'] - 100)**2)**0.5) / (((test['x_1'] - 100)**2 + (test['y_1']-50)**2)**0.5))).astype(float)), 0))



test['Legioner'] = np.where((test['league'] == "IT") & (test['passportArea'] == "Italy"), 0,

                            np.where((test['league'] == "SP") & (test['passportArea'] == "Spain"), 0,

                                    np.where((test['league'] == "GE") & (test['passportArea'] == "Germany"), 0,

                                            np.where((test['league'] == "FR") & (test['passportArea'] == "France"), 0,

                                                    np.where((test['league'] == "EN") & (test['passportArea'] == "England"), 0, 1)))))
#df.drop_duplicates(['teamId', 'matchId', 'shot']).groupby(['teamId', 'matchId'])['shot'].mean()

#df['shot'] = df.iloc[:,3]



#df.groupby(['matchId'])['shot'].transform(lambda x: x.unique().mean())

df = pd.DataFrame(train.groupby(['teamId', 'matchId']).size().reset_index())



df1 = pd.DataFrame(df.groupby(['teamId'])[0].agg(lambda x: x.unique().mean()))

df1.rename(columns = {0:'team_shots'}, inplace=True)

train = train.merge(df1, how = 'left', left_on='teamId', right_on='teamId')
df = pd.DataFrame(test.groupby(['teamId', 'matchId']).size().reset_index())



df1 = pd.DataFrame(df.groupby(['teamId'])[0].agg(lambda x: x.unique().mean()))

df1.rename(columns = {0:'team_shots'}, inplace=True)

test = test.merge(df1, how = 'left', left_on='teamId', right_on='teamId')
df = pd.DataFrame(train.groupby(['matchId', 'teamId']).size().reset_index())



df1 = pd.DataFrame(df.groupby(['matchId'])[0].agg(lambda x: x.unique().mean()))

df1.rename(columns = {0:'match_shots'}, inplace=True)

train = train.merge(df1, how = 'left', left_on='matchId', right_on='matchId')
df = pd.DataFrame(test.groupby(['matchId', 'teamId']).size().reset_index())



df1 = pd.DataFrame(df.groupby(['matchId'])[0].agg(lambda x: x.unique().mean()))

df1.rename(columns = {0:'match_shots'}, inplace=True)

test = test.merge(df1, how = 'left', left_on='matchId', right_on='matchId')
df = pd.DataFrame(train.groupby(['playerId', 'matchId']).size().reset_index())



df1 = pd.DataFrame(df.groupby(['playerId'])[0].agg(lambda x: x.unique().mean()))

df1.rename(columns = {0:'player_shots'}, inplace=True)

train = train.merge(df1, how = 'left', left_on='playerId', right_on='playerId')
df = pd.DataFrame(test.groupby(['playerId', 'matchId']).size().reset_index())



df1 = pd.DataFrame(df.groupby(['playerId'])[0].agg(lambda x: x.unique().mean()))

df1.rename(columns = {0:'player_shots'}, inplace=True)

test = test.merge(df1, how = 'left', left_on='playerId', right_on='playerId')
train.drop(columns = 'teamId', inplace = True)

test.drop(columns = 'teamId', inplace = True)



train.drop(columns = 'matchId', inplace = True)

test.drop(columns = 'matchId', inplace = True)



train.drop(columns = 'birth', inplace = True)

test.drop(columns = 'birth', inplace = True)



train.drop(columns = 'shot_id', inplace = True)

test.drop(columns = 'shot_id', inplace = True)



train.drop(columns = 'playerId', inplace = True)

test.drop(columns = 'playerId', inplace = True)
train.select_dtypes(include = ['object'])
train.corr()
X = train.select_dtypes(exclude = ['object']).fillna(-1)

X.drop(columns = 'is_goal', inplace = True)

y_tr = train['is_goal'].values

X_tr = X.values

test_obrabot = test.select_dtypes(exclude = ['object']).fillna(-1)
from sklearn.model_selection import train_test_split



X_tr, X_test, y_tr, y_test = train_test_split(X_tr, y_tr, test_size = 0.2, random_state = 0)

X_tr.shape, X_test.shape, y_tr.shape, y_test.shape



from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb



xg_class = xgb.XGBClassifier(

    learning_rate=0.02, 

    max_delta_step=0, 

    max_depth=3, #10,

    min_child_weight=5, #0.1, 

    missing=None, 

    n_estimators=250, 

    nthread=4,

    objective='binary:logistic', 

    reg_alpha=0.01, 

    reg_lambda = 0.01,

    scale_pos_weight=1, 

    seed=0, 

    #silent=True, 

    subsample=1) #0.9)



xg_fit=xg_class.fit(X_tr, y_tr)



roc_auc_score(y_test, xg_class.predict_proba(X_test)[:,1]), roc_auc_score(y_tr, xg_class.predict_proba(X_tr)[:,1])

#(0.7678727858633386, 0.9981305023093079)

#(0.773255976456168, 0.7846403099711974)



#(0.7735184381009774, 0.784546541135321)

from sklearn.model_selection import cross_val_score



np.mean(cross_val_score(xg_class, X_tr, y_tr, cv = 10, scoring= 'roc_auc')) #'roc_auc') #, #scoring = roc_auc_score)
pd.DataFrame({

    'variable': X.columns,

    'importance': xg_fit.feature_importances_

}).sort_values('importance', ascending=False)
from sklearn.pipeline import Pipeline

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



xg_class_grid = xgb.XGBClassifier (random_state = 0, objective='binary:logistic',

                   nthread=4, missing = None) # silent=True,

param_grid = {

        'param__min_child_weight': [8], #[1, 3, 5], #[5], 555

        'param__gamma': [3], #[1, 3, 5], #[5],

        'param__subsample': [0.8], #[0,5, 0.6, 0.8, 0.9], #[1], 0.8

        'param__colsample_bytree': [0.7], #[0.8],

        'param__max_depth': [2], #[2, 3, 4, 5], #333

        'param__n_estimators': [600],

        'param__learning_rate' : [0.02], #0.01, 0.03],

        'param__max_delta_step' : [0],

        #'param__missing': ['None'],

        #'param__ nthread=4,

        'param__reg_alpha' : [0.04], #[0.01, 0.02, 0.03], 

        'param__reg_lambda' : [0.04], #[0.01, 0.02, 0.03],

        'param__scale_pos_weight': [1],

        'param__seed' : [0],   

        }

    

pipeline = Pipeline([ ('param', xg_class_grid)])



grid = GridSearchCV(pipeline, cv = 10, param_grid = param_grid, n_jobs=-1, 

                    scoring='roc_auc')

grid.fit(X_tr, y_tr)



print('Best parameters found by grid search are:', grid.best_params_)

print('Best score found by grid search is:', grid.best_score_)

#0.7733440037081725

#(0.7623493089753883, 0.8047825947514051)





roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]), roc_auc_score(y_tr, grid.predict_proba(X_tr)[:,1])
#0.7730951529235991

#(0.7664714523012375, 0.7891694530375316)
#Best score found by grid search is: 0.7738371273389062

 #   (0.7663870702191697, 0.7892375348307237)



#grid.best_estimator_.named_steps["pipeline"].feature_importances_ #named_steps["pipeline"].



#grid.best_estimator_.feature_importances_
#pd.DataFrame({

#    'variable': X.columns,

#    'importance': grid.best_estimator_.named_steps["pipeline"].feature_importances_ #xg_fit

#}).sort_values('importance', ascending=False)
roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]), roc_auc_score(y_tr, grid.predict_proba(X_tr)[:,1])
roc_auc_score(y_test, xg_class.predict_proba(X_test)[:,1]), roc_auc_score(y_tr, xg_class.predict_proba(X_tr)[:,1])
print(f1_score(y_test, grid.predict(X_test)), "TRAIN:",f1_score(y_tr, grid.predict(X_tr)))
print(f1_score(y_test, xg_class.predict(X_test)), "TRAIN:",f1_score(y_tr, xg_class.predict(X_tr)))
{i: f1_score(y_test, (grid.predict_proba(X_test)[:,1] > i).astype('int')) for i in np.array(list(range(10, 50)))/100}
{i: f1_score(y_test, (xg_class.predict_proba(X_test)[:,1] > i).astype('int')) for i in np.array(list(range(10, 50)))/100}
treshold_grid = 0.18

print(f1_score(y_test, (grid.predict_proba(X_test)[:,1] > treshold_grid).astype('int')), 

      "TRAIN:",f1_score(y_tr, (grid.predict_proba(X_tr)[:,1] > treshold_grid).astype('int')))
#0.40574712643678157 TRAIN: 0.449438202247191
treshold_model = 0.18

print(f1_score(y_test, (xg_class.predict_proba(X_test)[:,1] > treshold_model).astype('int')), 

      "TRAIN:",f1_score(y_tr, (xg_class.predict_proba(X_tr)[:,1] > treshold_model).astype('int')))
import lightgbm



train_data = lightgbm.Dataset(X_tr, label = y_tr)

test_data = lightgbm.Dataset(X_test, label = y_test)



parameters = {

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 30,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.9,

    'bagging_freq': 5,

    'learning_rate': 0.2,

    'verbose': 0

}



lgb_model = lightgbm.train(parameters,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=5000,

                       early_stopping_rounds=100)
pd.DataFrame({

    'variable': X.columns,

    'importance': lgb_model.feature_importance(importance_type="gain")

}).sort_values('importance', ascending=False)
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from sklearn.model_selection import GridSearchCV, ParameterGrid

from sklearn.metrics import (roc_curve, auc, accuracy_score)



params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': True,

    'metric_freq': 1,

    'is_training_metric': True,

    'max_bin': 255,

    'learning_rate': 0.02,

    'num_leaves': 63,

    'lambda_l1': 0,

    'lambda_l2': 0,

    'tree_learner': 'serial',

    'feature_fraction': 0.8,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'min_data_in_leaf': 50,

    'min_sum_hessian_in_leaf': 5,

    'num_machines': 1,

    'local_listen_port': 12400,

    'verbose': 0,

    'min_child_samples': 20,

    'min_child_weight': 0.001,

    'min_split_gain': 0.0,

    'colsample_bytree': 1.0,

    'reg_alpha': 0.0,

    'reg_lambda': 0.0

}



gridParams = {

    #'max_depth': [7, 8, 10, 11, 12],

    'lambda_l1': [6], #[3, 4, 5, 10, 11, 12, 13], #[11],

    'lambda_l2': [4], #[0, 1, 2, 3], #[2],

    'num_leaves': [6], #[8, 9, 10, 11, 15, 20], #20],

    'feature_fraction': [0.3], #[0.2, 0.3, 0.4, 0.5], #, 0.4],

    'bagging_fraction': [0.1], #[0.4, 0.5, 0.6, 0.7] #, 0.4]

    'learning_rate': [0.02],

    'min_child_samples': [15],

    'min_child_weight': [0.001],

    'colsample_bytree': [0.7],

    'min_data_in_leaf': [50],

    'min_sum_hessian_in_leaf': [4],

}



mdl = lgb.LGBMClassifier(

    n_estimators = 1000,

    task = params['task'],

    metric = params['metric'],

    metric_freq = params['metric_freq'],

    is_training_metric = params['is_training_metric'],

    max_bin = params['max_bin'],

    feature_fraction = params['feature_fraction'],

    bagging_fraction = params['bagging_fraction'],

    n_jobs = -1

)



scoring = {'AUC': 'roc_auc'}



grid_lgb = GridSearchCV(mdl, gridParams, verbose=2, cv=10, 

                    scoring=scoring, n_jobs=-1, refit='AUC')



grid_lgb.fit(X_tr, y_tr)



print('Best parameters found by grid search are:', grid_lgb.best_params_)

print('Best score found by grid search is:', grid_lgb.best_score_)



print("Light_lgb ", roc_auc_score(y_test, grid_lgb.predict_proba(X_test)[:,1]), roc_auc_score(y_tr, grid_lgb.predict_proba(X_tr)[:,1]))
#Best score found by grid search is: 0.7738790956929777

#Light_lgb  0.7623419714030347 0.8008642797612084
pd.DataFrame({

    'variable': X.columns,

    'importance': grid_lgb.best_estimator_.feature_importances_ 

}).sort_values('importance', ascending=False)
print(f1_score(y_test, grid_lgb.predict(X_test)), "TRAIN:",f1_score(y_tr, grid_lgb.predict(X_tr)))
{i: f1_score(y_test, (grid_lgb.predict_proba(X_test)[:,1] > i).astype('int')) for i in np.array(list(range(10, 50)))/100}
#from catboost import CatBoostClassifier



#cat_model = CatBoostClassifier(

#    eval_metric='AUC',

#    use_best_model=True,

#    random_seed=42, iterations=2000)



#cat_model.fit(X_tr, y_tr, eval_set = (X_test, y_test))
#pd.DataFrame({

#    'variable': X.columns,

#    'importance': cat_model.feature_importances_

#}).sort_values('importance', ascending=False)
treshold_lgb = 0.19

print("F1_score Light_lgb ", f1_score(y_test, (grid_lgb.predict_proba(X_test)[:,1] > treshold_lgb).astype('int')), 

      "TRAIN:",f1_score(y_tr, (grid_lgb.predict_proba(X_tr)[:,1] > treshold_lgb).astype('int')))



treshold_grid = 0.18

print("F1_score XG_Boost ", f1_score(y_test, (grid.predict_proba(X_test)[:,1] > treshold_grid).astype('int')), 

      "TRAIN:",f1_score(y_tr, (grid.predict_proba(X_tr)[:,1] > treshold_grid).astype('int')))
#XG_Boost  0.7633594481027485 0.8074625130201806  

#XG_Boost  0.7623430196276566 0.8064790041755092 last

#Light_lgb  0.7587773088719636 0.826900746188638  

#Light_lgb  0.7589380366473304 0.8267665843679851 last
print("XG_Boost ", roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]), roc_auc_score(y_tr, grid.predict_proba(X_tr)[:,1]))

print("Light_lgb ", roc_auc_score(y_test, grid_lgb.predict_proba(X_test)[:,1]), roc_auc_score(y_tr, grid_lgb.predict_proba(X_tr)[:,1]))
print("Roc_Auc for XG_Boost+Light_lgb ", roc_auc_score(y_test, (grid.predict_proba(X_test)[:,1] * 0.55 + grid_lgb.predict(X_test) * 0.45)))
#np.around(xg_class.predict_proba(X_test)[:,1]).astype('int').sum()

#sub = np.where(xg_class.predict_proba(test_obrabot)[:,1] > 0.17, 1, 0)

sub = (xg_class.predict_proba(test_obrabot)[:,1] > treshold_model).astype('int')

sub_grid = (grid.predict_proba(test_obrabot)[:,1] > treshold_grid).astype('int')

sub_lgb = (grid_lgb.predict_proba(test_obrabot)[:,1] > treshold_lgb).astype('int')
sub.sum(), sub_grid.sum(), sub_lgb.sum()
goal_submission['is_goal'] = sub_grid

goal_submission.to_csv('submission1.csv', index = False)