#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQeyIuhXNV4WfW7V8AEQJEdI592V-PwaGQbR928Obn5z0AEVZPT&usqp=CAU',width=400,height=400)
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
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/covid19-open-datasets-for-brazil/Flu-Like Syndrome/dados-nacional_01_06.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'dados-nacional_01_06.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.isnull().sum()
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
for col in ('evolucaoCaso', 'dataEncerramento', 'cbo', 'dataNotificacao'):

    df[col] = df[col].fillna('None')
for col in ('condicoes', 'resultadoTeste', 'tipoTeste', 'classificacaoFinal', 'dataTeste', 'bairro', 'estadoTeste'):

    df[col] = df[col].fillna(df[col].mode()[0])
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(categorical_cols)
from sklearn.preprocessing import LabelEncoder

categorical_col = ('profissionalSaude', 'estadoTeste', 'tipoTeste', 'resultadoTeste', 'paisOrigem', 'sexo', 'estado', 'origem', 'estadoNotificacao', 'excluido', 'validado', 'evolucaoCaso', 'classificacaoFinal')

        

        

for col in categorical_col:

    label = LabelEncoder() 

    label.fit(list(df[col].values)) 

    df[col] = label.transform(list(df[col].values))



print('Shape all_data: {}'.format(df.shape))
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
# Save target value for later

y = df.resultadoTeste.values



# In order to make imputing easier, we combine train and test data

df.drop(['resultadoTeste'], axis=1, inplace=True)

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

#submission['Id'] = test_Id

#submission['resultadoTeste'] = final_predictions

#submission.to_csv('house_pricing_submission.csv',index=False)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR4VMlKNr5b886C0n7yTB-XHxufMF6P7jlpBgSiPhpIRYnHgEGZ&usqp=CAU',width=400,height=400)