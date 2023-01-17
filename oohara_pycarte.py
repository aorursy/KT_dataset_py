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
!pip install pycaret -q
!pip install optuna -q
def score(x1, x2):
    return (10*x1)**2 + x2**2

import optuna
optuna.logging.set_verbosity(30)

def objective(trial):
    hyperparameter1 = trial.suggest_uniform('hyperparameter1', -10, 10)
    hyperparameter2 = trial.suggest_uniform('hyperparameter2', -10, 10)
    return score(hyperparameter1, hyperparameter2)

study = optuna.create_study()
study.optimize(objective, n_trials=100,)
print(study.best_params)
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_slice(study, params=['hyperparameter1',"hyperparameter2"])
optuna.visualization.plot_contour(study, params=['hyperparameter1',"hyperparameter2"])
from pycaret.datasets import get_data
dataset = get_data('credit').drop_duplicates()
dataset.columns
data = dataset.sample(frac=0.3, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))
X = data.drop("default",axis=1)
y = data["default"]
X_test = data_unseen.drop("default",axis=1)
y_test = data_unseen["default"]
train_data = lgb.Dataset(X, y)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
params = {}
model = lgb.train(params, train_data)
y_predict = model.predict(X_test)
print(f"auc: {roc_auc_score(y_test, y_predict)}")
from sklearn.model_selection import train_test_split
def objective(trial):
    param = {
        # 'objective': trial.suggest_categorical("objective",  ['RMSE', 'MAE']),
        # 'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'feature_fraction': trial.suggest_discrete_uniform('feature_fraction', 0.4, 1.0, 0.1),
        # 'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
    }

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
    train_data = lgb.Dataset(X_train, y_train)
    model = lgb.train(param, train_data)
    y_predict = model.predict(X_valid)
    return roc_auc_score(y_valid,y_predict)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
model = lgb.train(study.best_trial.params,train_data)
y_predict = model.predict(X_test)
print(f"auc: {roc_auc_score(y_test, y_predict)}")
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_contour(study, params=['num_leaves',"max_depth"])
from pycaret.classification import *

exp_clf102 = setup(data = data, target = 'default', session_id=123,
#                   normalize = True, 
#                   transformation = True, 
#                   ignore_low_variance = True,
#                   bin_numeric_features = ['LIMIT_BAL', 'AGE'],
                  )
top5 = compare_models(n_select=5, fold=5,sort="AUC")
for model in top5:
    print(model)
lightbgm_model = top5[3]
lightbgm_model = create_model('lightgbm', fold = 5)
predict_model(lightbgm_model);
tuned_lgb = tune_model(lightbgm_model, optimize='AUC',fold=5, n_iter=20,choose_better=True)
predict_model(tuned_lgb);
final_lgb = finalize_model(tuned_lgb)
predicted = predict_model(final_lgb,data_unseen)
predicted.head()
roc_auc_score(predicted["default"],predicted["Score"])
tuned_top5 = [tune_model(model, optimize='AUC',fold=5,choose_better=True) for model in top5]
for model in tuned_top5:
    predict_model(model)
final_cat = finalize_model(tuned_top5[0])
predicted = predict_model(final_cat,data_unseen)
roc_auc_score(predicted["default"],predicted["Score"])
plot_model(tuned_lgb, plot='calibration')
calibrated_lgb = calibrate_model(tuned_lgb,fold=5)
plot_model(calibrated_lgb, plot='calibration')
predict_model(calibrated_lgb);
final_clb_lgb = finalize_model(calibrated_lgb)
predicted = predict_model(final_clb_lgb,data_unseen)
roc_auc_score(predicted["default"],predicted["Score"])
