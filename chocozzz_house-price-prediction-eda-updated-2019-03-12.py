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
df_train.head()
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm



train_columns = [c for c in df_train.columns if c not in ['id','price','per_price']]



model = sm.OLS(df_train['price'].values, df_train[train_columns])

result = model.fit()

print(result.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()



vif["Features"] = df_train.columns

vif["VIF Values"] = [variance_inflation_factor(

    df_train.values, i) for i in range(df_train.shape[1])]



vif.sort_values(by='VIF Values',ascending=False)
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import RidgeCV



param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.015,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4950}



y_reg = df_train['price']



#prepare fit model with cross-validation

folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][train_columns], label=y_reg.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][train_columns], label=y_reg.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(df_test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    

cv = np.sqrt(mean_squared_error(oof, y_reg))

print(cv)
cv1 = np.sqrt(mean_squared_error(np.expm1(oof), np.expm1(y_reg)))

print(cv1)
##plot the feature importance

cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
df_oof = pd.DataFrame(oof)

df_y_reg = pd.DataFrame(y_reg)



data = pd.concat([df_oof, df_y_reg], axis=1)

data.columns = ['oof','y_reg']

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='oof', y="y_reg", data=data)
data = pd.concat([df_oof, df_y_reg], axis=1)

data.columns = ['oof','y_reg']

data['error'] = data['y_reg'] - data['oof']

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='y_reg', y="error", data=data)
lgb1_oof = pd.DataFrame()

lgb1_oof['id'] = df_train.id

lgb1_oof['price'] = oof



lgb1_sub = pd.DataFrame()

lgb1_sub['id'] = df_test.id

lgb1_sub['price'] = predictions



cv1 = np.sqrt(mean_squared_error(np.expm1(oof), np.expm1(y_reg)))

print(cv1)
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import RidgeCV



param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.005,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 4950}



y_reg = np.expm1(df_train['price'])



#prepare fit model with cross-validation

folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][train_columns], label=y_reg.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][train_columns], label=y_reg.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(df_test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    

cv = np.sqrt(mean_squared_error(oof, y_reg))

print(cv)
lgb2_oof = pd.DataFrame()

lgb2_oof['id'] = df_train.id

lgb2_oof['price'] = oof



lgb2_sub = pd.DataFrame()

lgb2_sub['id'] = df_test.id

lgb2_sub['price'] = predictions



cv2 = np.sqrt(mean_squared_error(oof, y_reg))

print(cv2)
# code : https://www.kaggle.com/karell/kakr-2nd-house-price-xgb-starter-109145

import xgboost as xgb



xgb_params = {

    'eta': 0.01,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



log_y_reg = np.log1p(y_reg)

print('Transform DMatrix...')

dtrain = xgb.DMatrix(df_train[train_columns], log_y_reg)

dtest = xgb.DMatrix(df_test[train_columns])



print('Start Cross Validation...')



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=50,verbose_eval=500, show_stdv=False)

print('best num_boost_rounds = ', len(cv_output))

rounds = len(cv_output)



model = xgb.train(xgb_params, dtrain, num_boost_round = rounds)

preds = model.predict(dtest)



xgb1_sub = df_test[['id']]

xgb1_sub['price'] = preds
# code : https://www.kaggle.com/karell/kakr-2nd-house-price-xgb-starter-109145

import xgboost as xgb



xgb_params = {

    'eta': 0.01,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



print('Transform DMatrix...')

dtrain = xgb.DMatrix(df_train[train_columns], y_reg)

dtest = xgb.DMatrix(df_test[train_columns])



print('Start Cross Validation...')



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=20000, early_stopping_rounds=50,verbose_eval=500, show_stdv=False)

print('best num_boost_rounds = ', len(cv_output))

rounds = len(cv_output)



model = xgb.train(xgb_params, dtrain, num_boost_round = rounds)

preds = model.predict(dtest)



xgb2_sub = df_test[['id']]

xgb2_sub['price'] = preds
lgb_oof = pd.merge(lgb1_oof,lgb2_oof,how='left',on='id')

lgb_train = df_train[['id','price']]

lgb_oof = pd.merge(lgb_oof,lgb_train,how='left',on='id')

lgb_oof.columns = ['id','price1','price2','price']



lgb_ensemble = (0.9*np.expm1(lgb_oof['price1']) + 0.1*lgb_oof['price2']).values

cv = np.sqrt(mean_squared_error(lgb_ensemble, np.expm1(lgb_oof['price']).values))

print(cv)
lgb_sub = pd.merge(lgb1_sub,lgb2_sub,how='left',on='id')

lgb_sub.columns = ['id','price1','price2']

lgb_sub['price'] = (0.9*np.expm1(lgb_sub['price1']) + 0.1*lgb_sub['price2'])

lgb_sub = lgb_sub[['id','price']]
xgb_sub = pd.merge(xgb1_sub,xgb2_sub,how='left',on='id')

xgb_sub.columns = ['id','price1','price2']

xgb_sub['price'] = (0.9*np.expm1(xgb_sub['price1']) + 0.1*xgb_sub['price2'])

xgb_sub = xgb_sub[['id','price']]
ensemble_sub = pd.merge(lgb_sub,xgb_sub,how='left',on='id')

ensemble_sub.columns = ['id','price1','price2']

ensemble_sub['price'] = 0.9*ensemble_sub['price1'] + 0.1*ensemble_sub['price2']

ensemble_sub = ensemble_sub[['id','price']]

ensemble_sub.to_csv("submission.csv",index=False)