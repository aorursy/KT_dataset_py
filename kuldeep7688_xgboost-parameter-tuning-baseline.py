# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import re

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

import pickle

from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, KFold



from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support, roc_auc_score)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")

test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")

print("Shape of the data is {}".format(df.shape))

print("Shape of the test data is {}".format(test.shape))
target_column = "hospital_death"



doubtful_columns = [

    "cirrhosis",

    "diabetes_mellitus",

    "immunosuppression",

    "hepatic_failure",

    "leukemia",

    "lymphoma",

    "solid_tumor_with_metastasis",

    "gcs_unable_apache"

]



cols_with_around_70_percent_zeros = [

    "intubated_apache", "ventilated_apache",

]



cols_with_diff_dist_in_test = [

    "hospital_id", "icu_id"

]



selected_columns = [

    'd1_spo2_max', 'd1_diasbp_max', 'd1_temp_min', 'h1_sysbp_max', 'gender', 'heart_rate_apache', 

    'weight', 'icu_stay_type', 'd1_mbp_max', 'h1_resprate_max', 'd1_heartrate_min', 'apache_post_operative', 'apache_4a_hospital_death_prob', 

    'd1_mbp_min', 'apache_4a_icu_death_prob', 'd1_sysbp_max', 'icu_type', 'apache_3j_bodysystem', 'h1_sysbp_min', 'h1_resprate_min', 'd1_resprate_max', 

    'h1_mbp_min', 'ethnicity', 'arf_apache', 'resprate_apache', 'map_apache', 'temp_apache', 'icu_admit_source', 'h1_spo2_min', 

    'd1_spo2_min', 'd1_resprate_min', 'h1_mbp_max', 'height', 'age', 'h1_diasbp_max', 'd1_sysbp_min',

    'pre_icu_los_days', 'd1_heartrate_max', 'd1_diasbp_min', 'apache_2_bodysystem', 'gcs_eyes_apache', 'apache_2_diagnosis', 

    'gcs_motor_apache', 'd1_temp_max', 'h1_spo2_max', 'h1_heartrate_max', 'bmi', 'd1_glucose_min', 

    'h1_heartrate_min', 'gcs_verbal_apache', 'apache_3j_diagnosis', 'd1_glucose_max', 'h1_diasbp_min'

]

print(f'Total number of diff. dist.  columns are {len(cols_with_diff_dist_in_test)}')

print(f'Total number of columns with 70% 0s are {len(cols_with_around_70_percent_zeros)}')

print(f'Total number of doubtful columns are {len(doubtful_columns)}')

print(f'Total number of selected columns are {len(selected_columns)}')
len(selected_columns) + len(cols_with_around_70_percent_zeros) + len(cols_with_diff_dist_in_test) + len(doubtful_columns)
continuous_columns = [

    'd1_spo2_max', 'd1_diasbp_max', 'd1_temp_min', 'h1_sysbp_max', 'heart_rate_apache', 

    'weight', 'd1_mbp_max', 'h1_resprate_max', 'd1_heartrate_min', 'apache_4a_hospital_death_prob', 

    'd1_mbp_min', 'apache_4a_icu_death_prob', 'd1_sysbp_max', 'h1_sysbp_min', 'h1_resprate_min', 'd1_resprate_max', 

    'h1_mbp_min', 'resprate_apache', 'map_apache', 'temp_apache', 'h1_spo2_min', 

    'd1_spo2_min', 'd1_resprate_min', 'h1_mbp_max', 'height', 'age', 'h1_diasbp_max', 'd1_sysbp_min',

    'pre_icu_los_days', 'd1_heartrate_max', 'd1_diasbp_min', 'gcs_eyes_apache', 

    'gcs_motor_apache', 'd1_temp_max', 'h1_spo2_max', 'h1_heartrate_max', 'bmi', 'd1_glucose_min', 

    'h1_heartrate_min', 'gcs_verbal_apache', 'd1_glucose_max', 'h1_diasbp_min'

]

binary_columns = [

    "apache_post_operative", "arf_apache", "cirrhosis", "diabetes_mellitus", "immunosuppression",

    "hepatic_failure", "leukemia", "lymphoma", "solid_tumor_with_metastasis", "gcs_unable_apache",

    "intubated_apache", "ventilated_apache"



]

categorical_columns = [

    'icu_stay_type', 'icu_type', "apache_3j_bodysystem", 'ethnicity', "gender",

    'icu_admit_source', "apache_2_bodysystem", 'apache_2_diagnosis', 'apache_3j_diagnosis', 

]

high_cardinality_columns = [

    "hospital_id", "icu_id"

]



print(len(continuous_columns) + len(binary_columns) + len(categorical_columns) + len(high_cardinality_columns))
columns_to_be_used = list(set(

    doubtful_columns + cols_with_around_70_percent_zeros + cols_with_diff_dist_in_test + selected_columns))

print(f'Total columns to be used initially are {len(columns_to_be_used)}')



categorical_columns = list(set(categorical_columns))

continuous_columns = list(set(continuous_columns))

binary_columns = list(set(binary_columns))

high_cardinality_columns = list(set(high_cardinality_columns))

print(f'Total categorical columns to be used initially are {len(categorical_columns)}')

print(f'Total continuous columns to be used initially are {len(continuous_columns)}')

print(f'Total binary_columns to be used initially are {len(binary_columns)}')

print(f'Total high_cardinality_columns to be used initially are {len(high_cardinality_columns)}')
df_train, Y_tr = df[columns_to_be_used], df[target_column]

df_test = test[columns_to_be_used]

print(df_train.shape, Y_tr.shape, df_test.shape)
# for categorical label encoding

cat_labenc_mapping = {

    col: LabelEncoder()

    for col in categorical_columns

}



for col in tqdm_notebook(categorical_columns):

    df_train[col] = df_train[col].astype('str')

    cat_labenc_mapping[col] = cat_labenc_mapping[col].fit(

        np.unique(df_train[col].unique().tolist() + df_test[col].unique().tolist())

    )

    df_train[col] = cat_labenc_mapping[col].transform(df_train[col]) 

        



for col in tqdm_notebook(categorical_columns):

    print()

    df_test[col] = df_test[col].astype('str')

    df_test[col] = cat_labenc_mapping[col].transform(df_test[col])
# imputing



# for categorical

cat_col2imputer_mapping = {

    col: SimpleImputer(strategy='most_frequent')

    for col in categorical_columns

}



# for continuous

cont_col2imputer_mapping = {

    col: SimpleImputer(strategy='median')

    for col in continuous_columns

}



# for binary 

bin_col2imputer_mapping = {

    col: SimpleImputer(strategy='most_frequent')

    for col in binary_columns

}



# for high cardinality 

hicard_col2imputer_mapping = {

    col: SimpleImputer(strategy='median')

    for col in high_cardinality_columns

}



all_imp_dicts = [cat_col2imputer_mapping, cont_col2imputer_mapping, bin_col2imputer_mapping,  hicard_col2imputer_mapping]



# fitting imputers

for imp_mapping_obj in tqdm_notebook(all_imp_dicts):

    for col, imp_object in imp_mapping_obj.items():

        data = df_train[col].values.reshape(-1, 1)

        imp_object.fit(data)



# transofrming imputed columns

# fitting imputers

for imp_mapping_obj in tqdm_notebook(all_imp_dicts):

    for col, imp_object in imp_mapping_obj.items():

        data = df_train[col].values.reshape(-1, 1)

        data = imp_object.transform(data)

        df_train[col] = list(data.reshape(-1,))



# inputing on test 

for imp_mapping_obj in tqdm_notebook(all_imp_dicts):

    for col, imp_object in imp_mapping_obj.items():

        data = df_test[col].values.reshape(-1, 1)

        data = imp_object.transform(data)

        df_test[col] = list(data.reshape(-1,))
# train_test split

X_train, X_eval, Y_train, Y_eval = train_test_split(df_train, Y_tr, test_size=0.15, stratify=Y_tr)

X_train.shape, X_eval.shape, Y_train.shape, Y_eval.shape
# tuning tree specific features

gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=Y_train)



fit_params_of_xgb = {

    "early_stopping_rounds":100, 

    "eval_metric" : 'auc', 

    "eval_set" : [(X_eval,Y_eval)],

    # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

    'verbose': 100,

}





# A parameter grid for XGBoost

params = {

    'booster': ["gbtree"],

    'learning_rate': [0.1],

    'n_estimators': range(100, 500, 100),

    'min_child_weight': [1],

    'gamma': [0],

    'subsample': [0.8],

    'colsample_bytree': [0.8],

    'max_depth': [5],

    "scale_pos_weight": [1]

}



xgb_estimator = XGBClassifier(

    objective='binary:logistic',

    # silent=True,

)



gsearch = GridSearchCV(

    estimator=xgb_estimator,

    param_grid=params,

    scoring='roc_auc',

    n_jobs=-1,

    cv=gkf, verbose=3

)



# gsearch = RandomizedSearchCV(

#     estimator=xgb_estimator,

#     param_distributions=params,

#     scoring='roc_auc',

#     n_jobs=-1,

#     cv=gkf, verbose=3

# )



xgb_model = gsearch.fit(X=X_train, y=Y_train, **fit_params_of_xgb)

gsearch.best_params_, gsearch.best_score_
gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=Y_train)



fit_params_of_xgb = {

    "early_stopping_rounds":100, 

    "eval_metric" : 'auc', 

    "eval_set" : [(X_eval,Y_eval)],

    # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

    'verbose': 100,

}





# A parameter grid for XGBoost

params = {

    'booster': ["gbtree"],

    'learning_rate': [0.1],

    'n_estimators': [300],

    'gamma': [0],

    'subsample': [0.8],

    'colsample_bytree': [0.8],

    "scale_pos_weight": [1],

    'max_depth':range(2, 7, 2),

    'min_child_weight':range(2, 8, 2)

}



xgb_estimator = XGBClassifier(

    objective='binary:logistic',

    silent=True,

)



gsearch = GridSearchCV(

    estimator=xgb_estimator,

    param_grid=params,

    scoring='roc_auc',

    n_jobs=-1,

    cv=gkf, verbose=3

)



xgb_model = gsearch.fit(X=X_train, y=Y_train, **fit_params_of_xgb)

gsearch.best_params_, gsearch.best_score_
gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=Y_train)



fit_params_of_xgb = {

    "early_stopping_rounds":100, 

    "eval_metric" : 'auc', 

    "eval_set" : [(X_eval,Y_eval)],

    # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

    'verbose': 100,

}





# A parameter grid for XGBoost

params = {

    'booster': ["gbtree"],

    'learning_rate': [0.1],

    'n_estimators': [300],

    'subsample': [0.8],

    'colsample_bytree': [0.8],

    "scale_pos_weight": [1],

    'max_depth':[4],

    'min_child_weight': [6],

    'gamma': [0, 0.01, 0.01]

}



xgb_estimator = XGBClassifier(

    objective='binary:logistic',

    silent=True,

)



gsearch = GridSearchCV(

    estimator=xgb_estimator,

    param_grid=params,

    scoring='roc_auc',

    n_jobs=-1,

    cv=gkf, verbose=3

)



# gsearch = RandomizedSearchCV(

#     estimator=xgb_estimator,

#     param_distributions=params,

#     scoring='roc_auc',

#     n_jobs=-1,

#     cv=gkf, verbose=3

# )



xgb_model = gsearch.fit(X=X_train, y=Y_train, **fit_params_of_xgb)

gsearch.best_params_, gsearch.best_score_
gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=Y_train)



fit_params_of_xgb = {

    "early_stopping_rounds":100, 

    "eval_metric" : 'auc', 

    "eval_set" : [(X_eval,Y_eval)],

    # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

    'verbose': 100,

}





# A parameter grid for XGBoost

params = {

    'booster': ["gbtree"],

    'learning_rate': [0.1],

    'n_estimators': [300],

    "scale_pos_weight": [1],

    'max_depth':[4],

    'min_child_weight': [6],

    'gamma': [0],

    'subsample': [i/ 10.0 for i in range(2, 5)],

    'colsample_bytree': [i/ 10.0 for i in range(8, 10)]

}



xgb_estimator = XGBClassifier(

    objective='binary:logistic',

    silent=True,

)



gsearch = GridSearchCV(

    estimator=xgb_estimator,

    param_grid=params,

    scoring='roc_auc',

    n_jobs=-1,

    cv=gkf, verbose=3

)



# gsearch = RandomizedSearchCV(

#     estimator=xgb_estimator,

#     param_distributions=params,

#     scoring='roc_auc',

#     n_jobs=-1,

#     cv=gkf, verbose=3

# )



xgb_model = gsearch.fit(X=X_train, y=Y_train, **fit_params_of_xgb)

gsearch.best_params_, gsearch.best_score_
gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=Y_train)



fit_params_of_xgb = {

    "early_stopping_rounds":100, 

    "eval_metric" : 'auc', 

    "eval_set" : [(X_eval,Y_eval)],

    # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

    'verbose': 100,

}





# A parameter grid for XGBoost

params = {

    'booster': ["gbtree"],

    'learning_rate': [0.1],

    'n_estimators': [300],

    "scale_pos_weight": [1],

    'max_depth':[4],

    'min_child_weight': [6],

    'gamma': [0],

    'subsample': [0.4],

    'colsample_bytree': [0.8],

    'reg_alpha': [1, 0.5, 0.1, 0.08]

}



xgb_estimator = XGBClassifier(

    objective='binary:logistic',

    silent=True,

)



gsearch = GridSearchCV(

    estimator=xgb_estimator,

    param_grid=params,

    scoring='roc_auc',

    n_jobs=-1,

    cv=gkf, verbose=3

)



# gsearch = RandomizedSearchCV(

#     estimator=xgb_estimator,

#     param_distributions=params,

#     scoring='roc_auc',

#     n_jobs=-1,

#     cv=gkf, verbose=3

# )



xgb_model = gsearch.fit(X=X_train, y=Y_train, **fit_params_of_xgb)

gsearch.best_params_, gsearch.best_score_
gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=Y_train)



fit_params_of_xgb = {

    "early_stopping_rounds":100, 

    "eval_metric" : 'auc', 

    "eval_set" : [(X_eval,Y_eval)],

    # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

    'verbose': 100,

}



# A parameter grid for XGBoost

params = {

    'booster': ["gbtree"],

    'learning_rate': [0.01],

    'n_estimators': range(1000, 6000, 1000),

    "scale_pos_weight": [1],

    'max_depth':[4],

    'min_child_weight': [6],

    'gamma': [0],

    'subsample': [0.4],

    'colsample_bytree': [0.8],

    'reg_alpha': [0.08]

}



xgb_estimator = XGBClassifier(

    objective='binary:logistic',

    silent=True,

)



gsearch = GridSearchCV(

    estimator=xgb_estimator,

    param_grid=params,

    scoring='roc_auc',

    n_jobs=-1,

    cv=gkf, verbose=3

)



xgb_model = gsearch.fit(X=X_train, y=Y_train, **fit_params_of_xgb)

gsearch.best_params_, gsearch.best_score_
params_for_fit = {

    "eval_metric":"auc", "eval_set": [(X_eval, Y_eval)],

    'early_stopping_rounds':500, 'verbose': 100

}

xgb_estimator = XGBClassifier(

    n_estimators=3000,

    objective='binary:logistic',

    booster="gbtree",

    learning_rate=0.01,

    scale_pos_weight=1,

    max_depth=4,

    min_child_weight=6,

    gamma=0,

    subsample=0.4,

    colsample_bytree=0.8,

    reg_alpha=0.08,

#         n_jobs=-1

)

xgb_estimator.fit(X=X_train, y=Y_train, **params_for_fit)