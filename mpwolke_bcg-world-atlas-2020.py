#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSgZ0EC568XWLHgFm2YxY4w93Nge6RxFk-a9R3buKDArH5zJyna&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRcLpYvZ1YpK0YV3WrGysQrYF3RLKPtSiCQt5tt-sxJxhGc0CLp&usqp=CAU',width=400,height=400)
df = pd.read_csv('../input/hackathon/BCG_world_atlas_data-2020.csv', encoding='ISO-8859-2')

df.head()
na_percent = (df.isnull().sum()/len(df))[(df.isnull().sum()/len(df))>0].sort_values(ascending=False)



missing_data = pd.DataFrame({'Missing Percentage':na_percent*100})

missing_data
na = (df.isnull().sum() / len(df)) * 100

na = na.drop(na[na == 0].index).sort_values(ascending=False)



f, ax = plt.subplots(figsize=(12,8))

sns.barplot(x=na.index, y=na)

plt.xticks(rotation='90')

plt.xlabel('Features', fontsize=15)

plt.title('Percentage Missing', fontsize=15)
for col in ('File name of non bcgatlas datasource', 'Definition of High-risk groups (if applicable) which receive BCG?', 'BCG Manufacturer', 'Timing of revaccination (BCG #2, #3, #4)', 'BCG Policy Last Year', 'BCG Policy First Year', 'Country Code (Mandatory field)'):

    df[col] = df[col].fillna('None')
for col in ('BCG Policy Link (Mandatory field)', 'Are/were revaccinations (boosters) recommended?', 'Is it mandatory for all children?', 'Vaccination Timing (age)'):

    df[col] = df[col].fillna(df[col].mode()[0])
#for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

 #   dataset[col] = dataset[col].fillna(0)

    

for col in ['BCG Strain' ,'Location of Administration of BCG Vaccine', 'BCG Supply Company', 'Additional Comments']:

    df[col] = df[col].fillna('None')
df["BCG Strain"] = df.groupby("BCG Supply Company")["BCG Strain"].transform(lambda x: x.fillna(x.median()))
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(categorical_cols)
print(numerical_cols)
from sklearn.preprocessing import LabelEncoder

categorical_col = ('Is it from bcgatlas.org (Mandatory field)', 'File name of non bcgatlas datasource', 'Is it mandatory for all children?', 'Are/were revaccinations (boosters) recommended?')

        

        

for col in categorical_col:

    label = LabelEncoder() 

    label.fit(list(df[col].values)) 

    df[col] = label.transform(list(df[col].values))



print('Shape all_data: {}'.format(df.shape))
plt.style.use('fivethirtyeight')

sns.countplot(df['Are/were revaccinations (boosters) recommended?'],linewidth=3,palette="Set2",edgecolor='black')

plt.show()
from scipy.stats import norm, skew

num_features = df.dtypes[df.dtypes != 'object'].index

skewed_features = df[num_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewness.head(15)
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
from sklearn.model_selection import train_test_split

# Hot-Encode Categorical features

df = pd.get_dummies(df) 



# Splitting dataset back into X and test data

X = df[:len(df)]

test = df[len(df):]



X.shape
df = df.rename(columns={'Is it mandatory for all children?':'mandatory'})
# Save target value for later

y = df.mandatory.values



# In order to make imputing easier, we combine train and test data

df.drop(['mandatory'], axis=1, inplace=True)

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
# Predict every model

lasso_pred = lasso.predict(test)

ridge_pred = ridge.predict(test)

elasticnet_pred = elasticnet.predict(test)

lightgbm_pred = lightgbm.predict(test)

xgboost_pred = xgboost.predict(test)
elasticnet_pred = elasticnet.predict(test)

# Combine predictions into final predictions

final_predictions = np.expm1((0.3*elasticnet_pred) + (0.3*lasso_pred) + (0.2*ridge_pred) + 

               (0.1*xgboost_pred) + (0.1*lightgbm_pred))
#submission = pd.DataFrame()

#submission['BCG Strain'] = test_BCG_Strain

#submission['mandatory'] = final_predictions

#submission.to_csv('hackathon/BCG_world_atlas_data-2020.csv',index=False)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)