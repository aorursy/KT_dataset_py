import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



import gc

import pickle

import time

from joblib import dump, load

from sklearn.metrics import mean_squared_error



from sklearn.linear_model import LinearRegression



from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

import xgboost as xgb

from xgboost import XGBRegressor
# This cell is only for displaying package versions. You can ignore it.



import pkg_resources

import types

def get_imports():

    for name, val in globals().items():

        if isinstance(val, types.ModuleType):

            # Split ensures you get root package, 

            # not just imported function

            name = val.__name__.split(".")[0]



        elif isinstance(val, type):

            name = val.__module__.split(".")[0]



        # Some packages are weird and have different

        # imported names vs. system names

        if name == "PIL":

            name = "Pillow"

        elif name == "sklearn":

            name = "scikit-learn"



        yield name

imports = list(set(get_imports()))



requirements = []

for m in pkg_resources.working_set:

    if m.project_name in imports and m.project_name!="pip":

        requirements.append((m.project_name, m.version))



for r in requirements:

    print("{}=={}".format(*r))
# loading the trained models

xgbm = load("../input/xgboost-model/xgb_model.joblib")

lgbm = load("../input/lightgbm-model/lightgbm_model.joblib")

rfm = load("../input/randomforest-model/rf_model.joblib")
fig, ax = plt.subplots(1,1,figsize=(12,8))

xgb.plot_importance(xgbm, ax=ax);
lgb.plot_importance(

    lgbm,  

    importance_type='gain', 

    figsize=(12,8));
feat_importances = pd.Series(rfm.feature_importances_, index=xgbm.get_booster().feature_names)

feat_importances.sort_values().plot(kind='barh', figsize=(12,8))
#XGBoost model

xgb_train  = pd.read_csv('../input/xgboost-model/xgb_y_train.csv').set_index('ID')

xgb_valid  = pd.read_csv('../input/xgboost-model/xgb_y_valid.csv').set_index('ID')

xgb_test  = pd.read_csv('../input/xgboost-model/xgb_submission.csv').set_index('ID')



# LightGBM model

lgb_train  = pd.read_csv('../input/lightgbm-model/gbm_y_train.csv').set_index('ID')

lgb_valid  = pd.read_csv('../input/lightgbm-model/gbm_y_valid.csv').set_index('ID')

lgb_test  = pd.read_csv('../input/lightgbm-model/gbm_y_test.csv').set_index('ID')



# RandomForest model

rf_train  = pd.read_csv('../input/randomforest-model/rf_y_train.csv').set_index('ID')

rf_valid  = pd.read_csv('../input/randomforest-model/rf_y_valid.csv').set_index('ID')

rf_test  = pd.read_csv('../input/randomforest-model/rf_y_test.csv').set_index('ID')



# loading test set

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')

# Adding the first level model predictions together

X_train = np.hstack([xgb_train.values, lgb_train.values, rf_train.values])

X_valid = np.hstack([xgb_valid.values, lgb_valid.values, rf_valid.values])

X_test = np.hstack([xgb_test.values, lgb_test.values, rf_test.values])
data = pd.read_pickle('../input/eda-preprocessing-feature-engineering/all_data.pkl')

# Dropping the first 6 months because they were used for lags

data = data[data.date_block_num > 5]



# Doing the time based train-val-test split

Y_train = data[data.date_block_num < 33]['item_cnt_month']

Y_valid = data[data.date_block_num == 33]['item_cnt_month']



del data

gc.collect()
start_time = time.time()

# Fitting on valid, because the first level models were not trained on it and this way we reduce overfitting



# Using liner regression as the second level model

lr = LinearRegression(n_jobs=-1, normalize=True)

lr.fit(X_valid, Y_valid)



print(f"Training took {time.time() - start_time} s")



start_time = time.time()

Y_valid_pred = lr.predict(X_valid).clip(0, 20)

print(f"Predicting on valid set took {time.time() - start_time} s")



print(f"VALID RMSE: {round(np.sqrt(mean_squared_error(Y_valid, Y_valid_pred)), 5)}")





# Saving the trained model to disk

dump(lr, 'stacking_model.joblib') 
start_time = time.time()

Y_test = lr.predict(X_test).clip(0, 20)

print(f"Predicting test set took {time.time() - start_time} s")



# Creating the submission



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('submission.csv', index=False)

weights = pd.Series(lr.coef_, index=["XGBoost", "LightGBM", "Random Forest"])

weights.sort_values().plot(kind='barh', figsize=(12,8))