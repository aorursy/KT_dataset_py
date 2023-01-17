# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
from bayes_opt import BayesianOptimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import roc_auc_score,mean_absolute_error,mean_squared_error
from mlxtend.regressor import StackingCVRegressor


from sklearn.model_selection import cross_val_score,KFold


# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
print(len(df_train))
print(len(df_test))
df_test
df_train
df_train.describe()
df_test.describe()
test_index=df_test['Unnamed: 0']
print(test_index)


df_train.info()
plt.figure(figsize=(25,30))

# Add title
plt.title("analysis")

# Heatmap 
sns.heatmap(df_train.corr(),  annot=True)
plt.figure(figsize=(25,30))

# Add title
plt.title("analysis")

# Heatmap 
sns.heatmap(df_test.corr(),  annot=True)
df_test.corr()
df_test.info()
count=0
counnt=0
for i in df_test['F4']:
    if i==1:
        count+=1
for i in df_train['F4']:
    if i==1:
        counnt+=1
print(count,counnt)

df_train.drop(['F1', 'F2'], axis = 1, inplace = True)

train_X = df_train.loc[:, 'F3':'F17']
train_y = df_train.loc[:, 'O/P']

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=43)

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, train_X, train_y,
                                    scoring="neg_mean_squared_error",
                                    cv=kfolds))
    return rmse
# space = {


#         'n_estimators':hp.choice('n_estimators', np.arange(400, 1000, 10, dtype=int)),

    
#         'gamma': hp.uniform ('gamma', 1,15),

#         'subsample':hp.quniform('subsample', 0.5, 0.9, 0.01),

#         'eta':hp.quniform('eta', 0.05, 0.5, 0.01),

#         'objective':'reg:squarederror',


#         'eval_metric': 'rmse',

#     }




# def score(params):

#     model = XGBRegressor(**params)

#     model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],

#               verbose=False, early_stopping_rounds=10,eval_metric='rmse')

#     Y_pred = model.predict(X_test)

#     score = np.sqrt(mean_squared_error(y_test, Y_pred)) 
# #     score=np.sqrt(-cross_val_score(model,train_X,train_y,cv=kfolds,scoring='neg_mean_squared_error'))

#     print(score)

#     return {'loss': score, 'status': STATUS_OK}    

# def optimize(trials, space):

#     best = fmin(score, space, algo=tpe.suggest, max_evals=100)

#     return best


# trials = Trials()

# best_params = optimize(trials, space)


# # Return the best parameters

# space_eval(space, best_params)

# rf = RandomForestRegressor(n_estimators=500)
xgbr=XGBRegressor( gamma= 11.2,eta=0.07, n_estimators= 440,subsample=0.7)
# xgbb=XGBRegressor()
score = cv_rmse(xgbr)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), )
# space = {


#         'n_estimators':hp.choice('n_estimators', np.arange(400, 1200, 10, dtype=int)),

    
#         'min_child_weight': hp.uniform ('min_child_weight', 1,30),
    
#         'max_depth':hp.choice('max_depth', np.arange(5, 13, 1, dtype=int)),

#         'subsample':hp.quniform('subsample', 0.3, 0.9, 0.01),

#         'learning_rate':hp.quniform('learning_rate', 0.05, 0.5, 0.01),
        
#         'bagging_fraction': hp.uniform('bagging_fraction',0.5, 1),

#         'eval_metric': 'rmse',

#     }
# def score(params):

#     model = LGBMRegressor(**params)

#     model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],

#               verbose=False, early_stopping_rounds=10,eval_metric='rmse')

#     Y_pred = model.predict(X_test)

#     score = np.sqrt(mean_squared_error(y_test, Y_pred)) 
# #     score=np.sqrt(-cross_val_score(model,train_X,train_y,cv=kfolds,scoring='neg_mean_squared_error'))

#     print(score)

#     return {'loss': score, 'status': STATUS_OK}    

# def optimize(trials, space):

#     best = fmin(score, space, algo=tpe.suggest, max_evals=100)

#     return best


# trials = Trials()

# best_params = optimize(trials, space)


# # Return the best parameters

# space_eval(space, best_params)
lgbm=LGBMRegressor(bagging_fraction=0.77,eval_metric='rmse',learning_rate=0.07,max_depth=12,min_child_weight=13.87,n_estimators=920,subsample=0.73)
score = cv_rmse(lgbm)
print("lgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), )
# space = {


#         'n_estimators':hp.choice('n_estimators', np.arange(400, 1000, 10, dtype=int)),
    
#         'max_depth':hp.choice('max_depth', np.arange(5, 13, 1, dtype=int)),

#         'subsample':hp.quniform('subsample', 0.3, 0.9, 0.01),

#         'learning_rate':hp.quniform('learning_rate', 0.05, 0.5, 0.01),

#     }
# def score(params):

#     model = GradientBoostingRegressor(**params)

#     model.fit(X_train, y_train)

#     Y_pred = model.predict(X_test)

#     score = np.sqrt(mean_squared_error(y_test, Y_pred)) 
# #     score=np.sqrt(-cross_val_score(model,train_X,train_y,cv=kfolds,scoring='neg_mean_squared_error'))

#     print(score)

#     return {'loss': score, 'status': STATUS_OK}    

# def optimize(trials, space):

#     best = fmin(score, space, algo=tpe.suggest, max_evals=100)

#     return best


# trials = Trials()

# best_params = optimize(trials, space)


# # Return the best parameters

# space_eval(space, best_params)
gbm=GradientBoostingRegressor(learning_rate=0.05,max_depth=8,n_estimators=710,subsample=0.49)
# score = cv_rmse(gbm)
# print("gbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), )
stack_gen = StackingCVRegressor(regressors=(gbm, xgbr, lgbm),
                                meta_regressor=xgbr,
                                use_features_in_secondary=True)

stack_gen_model = stack_gen.fit(np.array(train_X), np.array(train_y))
xgb_model=xgbr.fit(train_X, train_y)
gbr_model=gbm.fit(train_X, train_y)
lgbm_model=lgbm.fit(train_X, train_y)

def blend_models_predict(X=train_X):
    return ((0.175 * gbr_model.predict(X)) + (0.175 * xgb_model.predict(X)) + (0.25 * lgbm_model.predict(X)) + (0.4 * stack_gen_model.predict(np.array(X))))

print('RMSLE score on train data:')
print(rmsle(train_y, blend_models_predict(train_X)))
df_test = df_test.loc[:, 'F3':'F17']

# pred = xgb.predict(df_test)
result = pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(blend_models_predict(df_test))


result.to_csv("output.csv", index=False)