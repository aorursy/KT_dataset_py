# Loading packages

import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",df_train.shape)

print("test.csv. Shape: ",df_test.shape)
df_train.head()
#descriptive statistics summary

df_train['price'].describe()
#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
#skewness and kurtosis

print("Skewness: %f" % df_train['price'].skew())

print("Kurtosis: %f" % df_train['price'].kurt())
fig = plt.figure(figsize = (15,10))



fig.add_subplot(1,2,1)

res = stats.probplot(df_train['price'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(df_train['price']), plot=plt)
df_train['price'] = np.log1p(df_train['price'])

#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['price'])
# correlation이 높은 상위 10개의 heatmap

# continuous + sequential variables --> spearman

# abs는 반비례관계도 고려하기 위함

# https://www.kaggle.com/junoindatascience/let-s-eda-it 준호님이 수정해 준 코드로 사용하였습니다. 

import scipy as sp



cor_abs = abs(df_train.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='price').index # price과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(df_train[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_living15']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living15', y="price", data=data)
data = pd.concat([df_train['price'], df_train['sqft_above']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_above', y="price", data=data)
data = pd.concat([df_train['price'], df_train['bathrooms']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='bathrooms', y="price", data=data)
data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

f, ax = plt.subplots(figsize=(18, 6))

fig = sns.boxplot(x='bedrooms', y="price", data=data)
from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



import plotly.graph_objs as go



import time

import random



#https://www.kaggle.com/ashishpatel26/bird-eye-view-of-two-sigma-nn-approach

def mis_value_graph(data):  

    data = [

    go.Bar(

        x = data.columns,

        y = data.isnull().sum(),

        name = 'Counts of Missing value',

        textfont=dict(size=20),

        marker=dict(

        line=dict(

            color= generate_color(),

            #width= 2,

        ), opacity = 0.45

    )

    ),

    ]

    layout= go.Layout(

        title= '"Total Missing Value By Column"',

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='skin')

    

def generate_color():

    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))

    return color



df_all = pd.concat([df_train,df_test])

del df_all['price']

mis_value_graph(df_all)
### 유니크 갯수 계산

train_unique = []

columns = ['bedrooms','bathrooms','floors','waterfront','view','condition','grade']



for i in columns:

    train_unique.append(len(df_train[i].unique()))

unique_train = pd.DataFrame()

unique_train['Columns'] = columns

unique_train['Unique_value'] = train_unique



data = [

    go.Bar(

        x = unique_train['Columns'],

        y = unique_train['Unique_value'],

        name = 'Unique value in features',

        textfont=dict(size=20),

        marker=dict(

        line=dict(

            color= generate_color(),

            #width= 2,

        ), opacity = 0.45

    )

    ),

    ]

layout= go.Layout(

        title= "Unique Value By Column",

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='skin')
df_train['floors'].unique()
data = pd.concat([df_train['price'], df_train['sqft_living']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='sqft_living', y="price", data=data)
df_train.loc[df_train['sqft_living'] > 13000]
df_train = df_train.loc[df_train['id']!=8990]
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='grade', y="price", data=data)
df_train.loc[(df_train['price']>12) & (df_train['grade'] == 3)]
df_train.loc[(df_train['price']>14.7) & (df_train['grade'] == 8)]
df_train.loc[(df_train['price']>15.5) & (df_train['grade'] == 11)]
df_train = df_train.loc[df_train['id']!=456]

df_train = df_train.loc[df_train['id']!=2302]

df_train = df_train.loc[df_train['id']!=4123]

df_train = df_train.loc[df_train['id']!=7259]

df_train = df_train.loc[df_train['id']!=2777]
data = pd.concat([df_train['price'], df_train['bedrooms']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='bedrooms', y="price", data=data)
skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)
for df in [df_train,df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])
df_train.head()
for df in [df_train,df_test]:

    # 방의 전체 갯수 

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    

    # 거실의 비율 

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    

    df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']

    

    # 면적 대비 거실의 비율 

    df['sqft_ratio_1'] = df['sqft_living'] / df['sqft_total_size']

    

    df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15'] 

    

    # 재건축 여부 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    df['date'] = df['date'].astype('int')
df_train['per_price'] = df_train['price']/df_train['sqft_total_size']

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

df_train = pd.merge(df_train,zipcode_price,how='left',on='zipcode')

df_test = pd.merge(df_test,zipcode_price,how='left',on='zipcode')



for df in [df_train,df_test]:

    df['zipcode_mean'] = df['mean'] * df['sqft_total_size']

    df['zipcode_var'] = df['var'] * df['sqft_total_size']

    del df['mean']; del df['var']
train_columns = [c for c in df_train.columns if c not in ['id','price','per_price']]

X = df_train[train_columns]

y = df_train['price']

x_sub = df_test[train_columns]
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline, make_pipeline

from scipy.stats import skew

from sklearn.decomposition import PCA, KernelPCA

from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.svm import SVR, LinearSVR

from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge

from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor

from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
# define cross validation strategy

def rmse_cv(model,X,y):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))

    return rmse
models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),

          ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),XGBRegressor(), LGBMRegressor()]
names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay", "Xgb", "LGB"]

for name, model in zip(names, models):

    score = rmse_cv(model, X, y)

    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                

# svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)         
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006,

                                     tree_method = "hist")
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
# score = cv_rmse(ridge)

# print("RIDGE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



# score = cv_rmse(lasso)

# print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



# score = cv_rmse(elasticnet)

# print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



# score = cv_rmse(svr)

# print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(gbr)

print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(xgboost)

print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
print('START Fit')



print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))



# print('elasticnet')

# elastic_model_full_data = elasticnet.fit(X, y)



# print('Lasso')

# lasso_model_full_data = lasso.fit(X, y)



# print('Ridge')

# ridge_model_full_data = ridge.fit(X, y)



# print('Svr')

# svr_model_full_data = svr.fit(X, y)



print('GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)



print('xgboost')

xgb_model_full_data = xgboost.fit(X, y)



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
# def blend_models_predict(X):

#     return ((0.1 * elastic_model_full_data.predict(X)) + \

#             (0.05 * lasso_model_full_data.predict(X)) + \

#             (0.1 * ridge_model_full_data.predict(X)) + \

#             (0.1 * svr_model_full_data.predict(X)) + \

#             (0.1 * gbr_model_full_data.predict(X)) + \

#             (0.15 * xgb_model_full_data.predict(X)) + \

#             (0.1 * lgb_model_full_data.predict(X)) + \

#             (0.3 * stack_gen_model.predict(np.array(X))))
# print('RMSLE score on train data:')

# print(rmsle(y, blend_models_predict(X)))
def blend_models_predict(X):

    return ((1 * gbr_model_full_data.predict(X)) + \

            (0.0 * xgb_model_full_data.predict(X)) + \

            (0.0 * lgb_model_full_data.predict(X)) + \

            (0 * stack_gen_model.predict(np.array(X))))

# 0.11

# 0.14

# 0.16

# 0.11
print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))
winsor = blend_models_predict(X)

winsor = np.floor(np.expm1(winsor))

winsor = pd.DataFrame(winsor, columns = ["Predict"])

# q1 = winsor['Predict'].quantile(0.005)

# q2 = winsor['Predict'].quantile(0.995)

# winsor['Predict'] = winsor['Predict'].apply(lambda x: x if x > q1 else x*0.77)

# winsor['Predict'] = winsor['Predict'].apply(lambda x: x if x < q2 else x*1.1)



pred_y = np.floor(np.expm1(y))

print('RMSLE score on train data:')

print(rmsle(pred_y, winsor))
def blend_models_predict(X):

    return ((1 * gbr_model_full_data.predict(X)) + \

            (0.0 * xgb_model_full_data.predict(X)) + \

            (0.0 * lgb_model_full_data.predict(X)) + \

            (0 * stack_gen_model.predict(np.array(X))))
prediction = np.expm1(blend_models_predict(x_sub))

prediction = pd.DataFrame(prediction, columns = ["Predict"])

sub = pd.DataFrame()

sub['id'] = df_test.id

sub['price'] = prediction.values

sub.to_csv("submission.csv", index = False)



prediction = np.expm1(blend_models_predict(x_sub))

prediction = pd.DataFrame(prediction, columns = ["Predict"])

q1 = prediction['Predict'].quantile(0.005)

q2 = prediction['Predict'].quantile(0.995)

prediction['Predict'] = prediction['Predict'].apply(lambda x: x if x > q1 else x*0.77)

prediction['Predict'] = prediction['Predict'].apply(lambda x: x if x < q2 else x*1.1)

sub = pd.DataFrame()

sub['id'] = df_test.id

sub['price'] = prediction.values

sub.to_csv("submission2.csv", index = False)
def blend_models_predict(X):

    return ((0.9 * gbr_model_full_data.predict(X)) + \

            (0.0 * xgb_model_full_data.predict(X)) + \

            (0.0 * lgb_model_full_data.predict(X)) + \

            (0.1 * stack_gen_model.predict(np.array(X))))
prediction = np.expm1(blend_models_predict(x_sub))

prediction = pd.DataFrame(prediction, columns = ["Predict"])

sub = pd.DataFrame()

sub['id'] = df_test.id

sub['price'] = prediction.values

sub.to_csv("submission3.csv", index = False)



prediction = np.expm1(blend_models_predict(x_sub))

prediction = pd.DataFrame(prediction, columns = ["Predict"])

q1 = prediction['Predict'].quantile(0.005)

q2 = prediction['Predict'].quantile(0.995)

prediction['Predict'] = prediction['Predict'].apply(lambda x: x if x > q1 else x*0.77)

prediction['Predict'] = prediction['Predict'].apply(lambda x: x if x < q2 else x*1.1)

sub = pd.DataFrame()

sub['id'] = df_test.id

sub['price'] = prediction.values

sub.to_csv("submission4.csv", index = False)
def blend_models_predict(X):

    return ((0.8 * gbr_model_full_data.predict(X)) + \

            (0.0 * xgb_model_full_data.predict(X)) + \

            (0.0 * lgb_model_full_data.predict(X)) + \

            (0.2 * stack_gen_model.predict(np.array(X))))
prediction = np.expm1(blend_models_predict(x_sub))

prediction = pd.DataFrame(prediction, columns = ["Predict"])

sub = pd.DataFrame()

sub['id'] = df_test.id

sub['price'] = prediction.values

sub.to_csv("submission5.csv", index = False)



prediction = np.expm1(blend_models_predict(x_sub))

prediction = pd.DataFrame(prediction, columns = ["Predict"])

q1 = prediction['Predict'].quantile(0.005)

q2 = prediction['Predict'].quantile(0.995)

prediction['Predict'] = prediction['Predict'].apply(lambda x: x if x > q1 else x*0.77)

prediction['Predict'] = prediction['Predict'].apply(lambda x: x if x < q2 else x*1.1)

sub = pd.DataFrame()

sub['id'] = df_test.id

sub['price'] = prediction.values

sub.to_csv("submission6.csv", index = False)