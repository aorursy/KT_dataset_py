#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
!pip install pycaret
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pycaret.regression import *

import numpy as np 

import pandas as pd 

from pandas_profiling import ProfileReport 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/task_2-Gemany_per_state_stats_20June2020.csv', encoding='utf8')

df.head()
report_df = ProfileReport(df)

report_df
index_int_float = ['Population']      



plt.figure(figsize=[20,12])

i = 1

for col in index_int_float :

    plt.subplot(4,10,i)

    sns.violinplot(x=col, data= df, orient='v')

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
index_str = ['Cases', 'Deaths', 'East/West','State_in_Germany_(English)', 'State_in_Germany_(German)']



plt.figure(figsize=[30,10])

i = 1

for col in index_str :

    plt.subplot(4,10,i)

    sns.scatterplot(x=col, y = 'Population_Density' ,data= df)

    sns.despine()

    i = i+1

plt.tight_layout()

plt.show()
int_features = ['Population']

        



float_features = [ ]



obj_features = ['Cases', 'Deaths', 'East/West','State_in_Germany_(English)', 'State_in_Germany_(German)']



exp_reg = setup(df, #Train Data

                target = 'Population_Density',  #Target

                categorical_features = obj_features, # Categorical Features

                numeric_features = int_features + float_features, # Numeric Features

                normalize = True, # Normalize Dataset

                remove_outliers = True, # Remove 5% Outliers

                remove_multicollinearity = True, # Remove Multicollinearity

                silent = True # Process Automation

               )
compare_models(blacklist = ['tr', 'catboost'], sort = 'RMSLE')
tuned_br = tune_model('br')

plot_model(tuned_br, plot = 'learning')
from sklearn.model_selection import train_test_split

# Hot-Encode Categorical features

df = pd.get_dummies(df) 



# Splitting dataset back into X and test data

X = df[:len(df)]

test = df[len(df):]



X.shape
# Save target value for later

y = df.Population.values



# In order to make imputing easier, we combine train and test data

df.drop(['Population'], axis=1, inplace=True)

df = pd.concat((df, test)).reset_index(drop=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.model_selection import KFold

# Indicate number of folds for cross validation

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)



# Parameters for models

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LassoCV

# Lasso Model

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas = alphas2, random_state = 42, cv=kfolds))



# Printing Lasso Score with Cross-Validation

lasso_score = cross_val_score(lasso, X, y, cv=kfolds, scoring='neg_mean_squared_error')

lasso_rmse = np.sqrt(-lasso_score.mean())

print("LASSO RMSE: ", lasso_rmse)

print("LASSO STD: ", lasso_score.std())
# Training Model for later

lasso.fit(X_train, y_train)
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas = alphas_alt, cv=kfolds))

ridge_score = cross_val_score(ridge, X, y, cv=kfolds, scoring='neg_mean_squared_error')

ridge_rmse =  np.sqrt(-ridge_score.mean())

# Printing out Ridge Score and STD

print("RIDGE RMSE: ", ridge_rmse)

print("RIDGE STD: ", ridge_score.std())
# Training Model for later

ridge.fit(X_train, y_train)
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))

elastic_score = cross_val_score(elasticnet, X, y, cv=kfolds, scoring='neg_mean_squared_error')

elastic_rmse =  np.sqrt(-elastic_score.mean())



# Printing out ElasticNet Score and STD

print("ELASTICNET RMSE: ", elastic_rmse)

print("ELASTICNET STD: ", elastic_score.std())
# Training Model for later

elasticnet.fit(X_train, y_train)
from lightgbm import LGBMRegressor

lightgbm = make_pipeline(RobustScaler(),

                        LGBMRegressor(objective='regression',num_leaves=5,

                                      learning_rate=0.05, n_estimators=720,

                                      max_bin = 55, bagging_fraction = 0.8,

                                      bagging_freq = 5, feature_fraction = 0.2319,

                                      feature_fraction_seed=9, bagging_seed=9,

                                      min_data_in_leaf =6, 

                                      min_sum_hessian_in_leaf = 11))



# Printing out LightGBM Score and STD

lightgbm_score = cross_val_score(lightgbm, X, y, cv=kfolds, scoring='neg_mean_squared_error')

lightgbm_rmse = np.sqrt(-lightgbm_score.mean())

print("LIGHTGBM RMSE: ", lightgbm_rmse)

print("LIGHTGBM STD: ", lightgbm_score.std())
# Training Model for later

lightgbm.fit(X_train, y_train)
from xgboost import XGBRegressor

xgboost = make_pipeline(RobustScaler(),

                        XGBRegressor(learning_rate =0.01, n_estimators=3460, 

                                     max_depth=3,min_child_weight=0 ,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,nthread=4,

                                     scale_pos_weight=1,seed=27, 

                                     reg_alpha=0.00006))



# Printing out XGBOOST Score and STD

xgboost_score = cross_val_score(xgboost, X, y, cv=kfolds, scoring='neg_mean_squared_error')

xgboost_rmse = np.sqrt(-xgboost_score.mean())

print("XGBOOST RMSE: ", xgboost_rmse)

print("XGBOOST STD: ", xgboost_score.std())
# Training Model for later

xgboost.fit(X_train, y_train)
results = pd.DataFrame({

    'Model':['Lasso',

            'Ridge',

            'ElasticNet',

            'LightGBM',

            'XGBOOST',

            ],

    'Score':[lasso_rmse,

             ridge_rmse,

             elastic_rmse,

             lightgbm_rmse,

             xgboost_rmse,

             

            ]})



sorted_result = results.sort_values(by='Score', ascending=True).reset_index(drop=True)

sorted_result
f, ax = plt.subplots(figsize=(14,8))

plt.xticks(rotation='90')

sns.barplot(x=sorted_result['Model'], y=sorted_result['Score'])

plt.xlabel('Model', fontsize=15)

plt.ylabel('Performance', fontsize=15)

plt.ylim(0.10, 0.12)

plt.title('RMSE', fontsize=15)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRsx2_7t2KW8B8EN09DXQ5Hp2_Kf132PgNMVA&usqp=CAU',width=400,height=400)