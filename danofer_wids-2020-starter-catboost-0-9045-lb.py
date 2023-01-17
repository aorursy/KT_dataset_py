import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from catboost import Pool, cv, CatBoostClassifier, CatBoostRegressor



from sklearn.metrics import mean_squared_error, classification_report

from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import train_test_split



# import xgboost

# import lightgbm as lgb

# from lightgbm import LGBMClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import gc

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import metrics

pd.set_option('max_rows', 300)

import re



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_columns', 300)

np.random.seed(566)

pd.set_option('display.max_rows', 200)

pd.set_option('display.width', 1000)

pd.set_option('display.float_format', '{:20,.2f}'.format)

pd.set_option('display.max_colwidth', -1)
TARGET_COL = "hospital_death"
df = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")

print(df.shape)

display(df.nunique())

df.head()
df.isna().sum()
df.describe()
test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")

print(test.shape)

display(test.nunique())

test.head()
test.isna().sum()
print([c for c in df.columns if 7<df[c].nunique()<800])

## 

# categorical_cols = ['hospital_id','apache_3j_bodysystem', 'apache_2_bodysystem',

# "hospital_admit_source","icu_id","ethnicity"]
## print non numeric columns : We may need to

## define them as categorical / encode as numeric with label encoder, depending on ml model used

print([c for c in df.columns if (1<df[c].nunique()) & (df[c].dtype != np.number)& (df[c].dtype != int) ])
categorical_cols =  ['hospital_id',

 'ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']



#['apache_3j_bodysystem', 'apache_2_bodysystem',

# "hospital_admit_source","icu_id","ethnicity"]
display(df[categorical_cols].dtypes)

display(df[categorical_cols].tail(3))

display(df[categorical_cols].isna().sum())
df[categorical_cols] = df[categorical_cols].fillna("")



# same transformation for test data

test[categorical_cols] = test[categorical_cols].fillna("")



df[categorical_cols].isna().sum()
## useful "hidden" function - df._get_numeric_data()  - returns only numeric columns from a pandas dataframe. Useful for scikit learn models! 



X_train = df.drop([TARGET_COL],axis=1)

y_train = df[TARGET_COL]
## catBoost Pool object

train_pool = Pool(data=X_train,label = y_train,cat_features=categorical_cols,

#                   baseline= X_train[""], ## 

#                   group_id = X_train['hospital_id']

                 )



### OPT/TODO:  do train test split for early stopping then add that as an eval pool object : 
model_basic = CatBoostClassifier(verbose=False,iterations=50)#,learning_rate=0.1, task_type="GPU",)

model_basic.fit(train_pool, plot=True,silent=True)

print(model_basic.get_best_score())
### hyperparameter tuning example grid for catboost : 

grid = {'learning_rate': [0.04, 0.1],

        'depth': [7, 11],

#         'l2_leaf_reg': [1, 3,9],

#        "iterations": [500],

       "custom_metric":['Logloss', 'AUC']}



model = CatBoostClassifier()



## can also do randomized search - more efficient typically, especially for large search space - `randomized_search`

grid_search_result = model.grid_search(grid, 

                                       train_pool,

                                       plot=True,

                                       refit = True, #  refit best model on all data

                                      partition_random_seed=42)



print(model.get_best_score())
print("best model params: \n",grid_search_result["params"])
feature_importances = model.get_feature_importance(train_pool)

feature_names = X_train.columns

for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

    if score > 0.05:

        print('{0}: {1:.2f}'.format(name, score))
import shap

shap.initjs()



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(train_pool)



# visualize the training set predictions

# SHAP plots for all the data is very slow, so we'll only do it for a sample. Taking the head instead of a random sample is dangerous! 

shap.force_plot(explainer.expected_value,shap_values[0,:400], X_train.iloc[0,:400])
# summarize the effects of all the features

shap.summary_plot(shap_values, X_train)
test[TARGET_COL] = model.predict(test.drop([TARGET_COL],axis=1),prediction_type='Probability')[:,1]
test[["encounter_id","hospital_death"]].to_csv("submission.csv",index=False)