import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# import lightgbm as lgb

# import xgboost as xgb

import catboost as cb

from catboost import Pool, cv, CatBoostClassifier

from sklearn import ensemble, preprocessing, tree, model_selection, feature_selection, pipeline, metrics

from sklearn.model_selection import StratifiedKFold



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

spending_score_dict = {"Low":0,"Average":1,"High":2 }
train = pd.read_csv('/kaggle/input/janatahack-customer-segmentation/Train.csv')

test = pd.read_csv('/kaggle/input/janatahack-customer-segmentation/Test.csv')
train["missings_count"] = train.isna().sum(axis=1)

test["missings_count"] = test.isna().sum(axis=1)



train["Spending_Score"] = train["Spending_Score"].map(spending_score_dict)

test["Spending_Score"] = test["Spending_Score"].map(spending_score_dict)



train["exp_div_age"] = train["Work_Experience"].div( train["Age"])

test["exp_div_age"] = test["Work_Experience"].div(test["Age"])



train["odd_experience"] = (train["Graduated"]!="Yes") & (train["Profession"].isin(['Healthcare',  'Engineer', 'Doctor', 'Lawyer',

       'Executive', 'Marketing'])).astype(int)

test["odd_experience"] = ((test["Graduated"]!="Yes") & (test["Profession"].isin(['Healthcare',  'Engineer', 'Doctor', 'Lawyer',

       'Executive', 'Marketing']))).astype(int)
train
train.columns
test
train.describe()
test.describe()
train["Profession"].value_counts(normalize=True)
train["Var_1"].value_counts(normalize=True)
categorical_cols = ['Gender', 'Ever_Married',  'Graduated', 'Profession','Var_1']
X_train = train.drop(["Segmentation"],axis=1)

X_train[categorical_cols] = X_train[categorical_cols].fillna('""')
X_train.isna().sum()
# !pip install --upgrade catboost
train_pool = Pool(

    X_train, 

    train["Segmentation"], 

    cat_features=categorical_cols,



)



# eval_pool = Pool(

#     X_test, 

#     y_test, 

#     cat_features=categorical_cols,

# )



catboost_params = {

    'iterations': 1800,

#     'learning_rate': 0.1,

#     "depth": 2,

#     'eval_metric': ['Logloss',"Accuracy"],

     "loss_function":'MultiClass',

    

    'task_type': 'GPU',

    'early_stopping_rounds': 15,

#     'use_best_model': True,

#     'verbose': 100,

    "silent":True,

#     "verbose": False,

}



# model = CatBoostClassifier(**catboost_params)





# model.fit(train_pool,plot=True)
###### scores = cv(train_pool,

#             catboost_params,

#             fold_count=4, 

#             plot="True")
model = CatBoostClassifier(**catboost_params)



# grid = {'learning_rate': [0.03, 0.07],

#         'depth': [4, 6, 10],

#         'l2_leaf_reg': [1, 3, 5, 7]}



# randomized_search_result = model.randomized_search(grid,

#                                                    train_pool,

# #                                                    X=train_data,

# #                                                    y=train_labels,

#                                                    n_iter=12,

#                                                    plot=True)





# randomized_search_result['params'] ### {'depth': 4, 'l2_leaf_reg': 1, 'learning_rate': 0.03}
# model = CatBoostClassifier(**catboost_params)



# grid = {'learning_rate': [0.02],

#         'depth': [2,4, 6,8],

#         'l2_leaf_reg': [ 1, 3],

#        "min_data_in_leaf":[1,3],

# #        "max_leaves":[31,61], ## cuda errors when searching this as well

#        "rsm":[1,0.8]

#        }



# grid_search_result = model.grid_search(grid,

#                                                    train_pool,

# #                                                    n_iter=12,

# #                                              cv=4,

#                                                    plot=True

#                                       )







# grid_search_result['params'] 



# #  {'rsm': 1,

# #  'min_data_in_leaf': 3,

# #  'depth': 6,

# #  'l2_leaf_reg': 3,

# #  'learning_rate': 0.02}


best_params = {'iterations': 1600,

    'learning_rate': 0.02,

'min_data_in_leaf': 2, 

 'depth': 6,

 'l2_leaf_reg': 3,

#     'eval_metric': ['Logloss',"Accuracy"],

     "loss_function":'MultiClass',

    'task_type': 'GPU',

    'early_stopping_rounds': 12,

#     'use_best_model': True,



    "silent":True,}



model = CatBoostClassifier(**best_params)



model.fit(train_pool)
test[categorical_cols] = test[categorical_cols].fillna('""')

preds = model.predict(test)

test["Segmentation"] = preds



display(test)

test[["ID","Segmentation"]].to_csv("output_preds_catboost_v3.csv",index=False)