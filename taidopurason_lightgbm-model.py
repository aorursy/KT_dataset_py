import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



import gc

import pickle

import time

from joblib import dump, load

import lightgbm as lgb



from sklearn.metrics import mean_squared_error
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
data = pd.read_pickle('../input/eda-preprocessing-feature-engineering/all_data.pkl')

# Dropping the first 6 months because they were used for lags

data = data[data.date_block_num > 5]

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')



# dropping some of the columns that didn't give any improvement

dropcols = [

            "item_cnt_month_lag_12",

            "item_cnt_month_lag_12_adv",

            "date_item_target_enc_lag_12",

            "date_shop_target_enc_lag_12",

            "date_city_target_enc_lag_1",

            "date_city_target_enc_lag_2",

            "date_city_target_enc_lag_3",

            "date_type_target_enc_lag_1",

            "date_subtype_target_enc_lag_1",

            "new_item_cat_avg_lag_1",

            "new_item_cat_avg_lag_2",

            "new_item_cat_avg_lag_3",

            "new_item_shop_cat_avg_lag_1",

            "new_item_shop_cat_avg_lag_2",

            "new_item_shop_cat_avg_lag_3",

           ]



# Doing the time based train-val-test split

X_train = data[data.date_block_num < 33].drop(['item_cnt_month']+dropcols, axis=1)

Y_train = data[data.date_block_num < 33]['item_cnt_month']

X_valid = data[data.date_block_num == 33].drop(['item_cnt_month']+dropcols, axis=1)

Y_valid = data[data.date_block_num == 33]['item_cnt_month']

X_test = data[data.date_block_num == 34].drop(['item_cnt_month']+dropcols, axis=1)



del data

gc.collect()
# Printing features

X_train.info()
start_time = time.time()



# https://www.kaggle.com/tylerssssss/feature-engineering-lightgbm

feature_name = X_train.columns.tolist()



params = {

    'objective': 'mse',

    'metric': 'rmse',

    'num_leaves': 2 ** 7 - 1,

    'learning_rate': 0.005,

    'feature_fraction': 0.73,

    'bagging_fraction': 0.75,

    'bagging_freq': 5,

    'seed': 0,

    'verbose': 1

}



feature_name_indexes = [ 

                        'country_part', 

                        'city_code',

                        'type_code',

                        'subtype_code',

                        'category_code',

                        'item_category_id', 

]



lgb_train = lgb.Dataset(X_train[feature_name], Y_train)

lgb_eval = lgb.Dataset(X_valid[feature_name], Y_valid, reference=lgb_train)



evals_result = {}

gbm = lgb.train(

        params, 

        lgb_train,

        num_boost_round=3000,

        valid_sets=(lgb_train, lgb_eval), 

        feature_name = feature_name,

        categorical_feature = feature_name_indexes,

        verbose_eval=5, 

        evals_result = evals_result,

        early_stopping_rounds = 100)



print(f"Training took {time.time() - start_time} s")



start_time = time.time()

Y_train_pred = gbm.predict(X_train).clip(0, 20)

print(f"Predicting on train set took {time.time() - start_time} s")



start_time = time.time()

Y_valid_pred = gbm.predict(X_valid).clip(0, 20)

print(f"Predicting on valid set took {time.time() - start_time} s")



print(f"TRAIN RMSE: {round(np.sqrt(mean_squared_error(Y_train, Y_train_pred)), 5)}, VALID RMSE: {round(np.sqrt(mean_squared_error(Y_valid, Y_valid_pred)), 5)}")





# Saving the trained model to disk

dump(gbm, 'lightgbm_model.joblib') 
start_time = time.time()

Y_test = gbm.predict(X_test).clip(0, 20)

print(f"Predicting test set took {time.time() - start_time} s")







submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('gbm_y_test.csv', index=False)



train_preds = pd.DataFrame({

    "ID": X_train.index, 

    "item_cnt_month": Y_train_pred

})

train_preds.to_csv('gbm_y_train.csv', index=False)



valid_preds = pd.DataFrame({

    "ID": X_valid.index, 

    "item_cnt_month": Y_valid_pred

})

valid_preds.to_csv('gbm_y_valid.csv', index=False)
