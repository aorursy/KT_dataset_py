import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import re



import matplotlib.pyplot as plt

import seaborn as sns



import plotly

plotly.offline.init_notebook_mode()

import plotly.graph_objs as go

import plotly.express as px



import plotly.figure_factory as ff

import cufflinks as cf



%matplotlib inline

sns.set_style("whitegrid")

sns.set_context("paper")

plt.style.use('seaborn')
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



#f = open('../input/house-prices-advanced-regression-techniques/data_description.txt','r')

#print(f.read())
df.sample(3)
df.drop('Id',axis=1, inplace=True)



category_features = list(df.dtypes[df.dtypes == 'object'].reset_index()['index'])

numerical_features = list(df.dtypes[df.dtypes != 'object'].reset_index()['index'])
null_df = df[category_features].isnull().sum().reset_index()

null_df.columns = ['Feature','Null_Values']

null_df = null_df.query('Null_Values > 0').sort_values(by='Null_Values')



fig  = px.bar(data_frame=null_df, x='Feature', y='Null_Values', template='seaborn')

fig.update_layout(width=800, height=300, title= {'text': "Categorical Features with Null values",

                                                'y':0.95,'x':0.5,

                                                'xanchor': 'center','yanchor': 'top'})
null_df = df[numerical_features].isnull().sum().reset_index()

null_df.columns = ['Feature','Null_Values']

null_df = null_df.query('Null_Values > 0').sort_values(by='Null_Values')



fig  = px.bar(data_frame=null_df, x='Feature', y='Null_Values', template='seaborn')

fig.update_layout(width=500, height=300, title= {'text': "Numerical Features with Null values",

                                                'y':0.95,'x':0.5,

                                                'xanchor': 'center','yanchor': 'top'})
# Creating a dataframe of columns having null values

null_df = df.isnull().sum().reset_index()

null_df.columns = ['Feature','Null_Values']

null_df = null_df.query('Null_Values > 0')



null_df_features = null_df['Feature']







# Function to fill Null values in Train/Test Dataset

def fill_na(features_series, dataset):

    for feature in list(features_series):

        # If datatype is Object

        if dataset[feature].dtype == 'O':

            # All the categorical features with null value mean 'No Present' as mentioned in the description

            dataset[feature] = dataset[feature].fillna('None')

        else:

            # If datatype is Float or Int

            dataset[feature] = dataset[feature].fillna(dataset[feature].median())



fill_na(null_df_features, df)
print(dict(df.isnull().sum()))
total_bsmt_calc = df['BsmtFinSF1'] + df['BsmtFinSF2'] +df['BsmtUnfSF']

fig = plt.figure(figsize=(5,5))

_ = sns.lineplot(total_bsmt_calc, df['TotalBsmtSF'])
df.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'], axis=1, inplace=True)
total_floor_sf = df['1stFlrSF'] + df['2ndFlrSF'] + df['LowQualFinSF']

fig = plt.figure(figsize=(5,5))

_ = sns.lineplot(total_floor_sf, df['GrLivArea'])
df.drop(['1stFlrSF','2ndFlrSF','LowQualFinSF'], axis=1, inplace=True)
total_porch_area = df[['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']].sum(axis=1)

df['TotalPorchSF'] = total_porch_area



df.drop(['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis=1, inplace=True)
df.head(5)
fig = plt.figure(figsize=(25,15))



ax1 = fig.add_subplot(421)

_ = sns.boxplot(data=df, y='LotArea',ax=ax1)

ax1.set_title('Lot Area', fontsize=30)



ax2 = fig.add_subplot(422)

_ = sns.boxplot(data=df, y='MasVnrArea',ax=ax2)

ax2.set_title('Masonry veneer type Area', fontsize=30)



ax3 = fig.add_subplot(423)

_ = sns.boxplot(data=df, y='TotalBsmtSF',ax=ax3)

ax3.set_title('TotalBsmtSF', fontsize=30)



ax4 = fig.add_subplot(424)

_ = sns.boxplot(data=df, y='GrLivArea',ax=ax4)

ax4.set_title('Ground Living Area', fontsize=30)



ax5 = fig.add_subplot(425)

_ = sns.boxplot(data=df, y='TotalPorchSF',ax=ax5)

ax5.set_title('Total Porch Area', fontsize=30)



ax6 = fig.add_subplot(426)

_ = sns.boxplot(data=df, y='GarageArea',ax=ax6)

ax6.set_title('Garage Area', fontsize=30)
numerical_features_new = list(df.dtypes[df.dtypes != 'object'].reset_index()['index'])



fig = plt.figure(figsize=(15,20))

ax = fig.gca()

_ = df[numerical_features_new].hist(ax=ax)
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline





scaler = MinMaxScaler(feature_range=(1, 2))

power = PowerTransformer(method='box-cox')

pipeline = Pipeline(steps=[('s', scaler),('p', power)])



df[numerical_features_new] = pipeline.fit_transform(df[numerical_features_new])
fig = plt.figure(figsize=(15,20))

ax = fig.gca()

_ = df[numerical_features_new].hist(ax=ax)
fig = plt.figure(figsize=(25,15))



ax1 = fig.add_subplot(421)

_ = sns.boxplot(data=df, y='LotArea',ax=ax1)

ax1.set_title('Lot Area', fontsize=30)



ax2 = fig.add_subplot(422)

_ = sns.boxplot(data=df, y='MasVnrArea',ax=ax2)

ax2.set_title('Masonry veneer type Area', fontsize=30)



ax3 = fig.add_subplot(423)

_ = sns.boxplot(data=df, y='TotalBsmtSF',ax=ax3)

ax3.set_title('TotalBsmtSF', fontsize=30)



ax4 = fig.add_subplot(424)

_ = sns.boxplot(data=df, y='GrLivArea',ax=ax4)

ax4.set_title('Ground Living Area', fontsize=30)



ax5 = fig.add_subplot(425)

_ = sns.boxplot(data=df, y='TotalPorchSF',ax=ax5)

ax5.set_title('Total Porch Area', fontsize=30)



ax6 = fig.add_subplot(426)

_ = sns.boxplot(data=df, y='GarageArea',ax=ax6)

ax6.set_title('Garage Area', fontsize=30)
X = df.drop(['SalePrice'], axis=1)

y = df['SalePrice']



X = pd.get_dummies(X,drop_first=True)
from sklearn.preprocessing import LabelEncoder



#label = LabelEncoder()



# Labelling categorical features

#for feature in category_features:

#    X[feature] = label.fit_transform(X[feature])
from sklearn.decomposition import PCA



pca = PCA()



X_PCA = pca.fit_transform(X)
display(pca.explained_variance_ratio_.cumsum())



n_features = range(pca.n_components_)

_ = plt.figure(figsize=(20,12))

_ = plt.bar(n_features, pca.explained_variance_)
pca = PCA(n_components=165)



X_PCA = pca.fit_transform(X)
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from sklearn.svm import SVR



from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from mlxtend.regressor import StackingCVRegressor
# Cross Validation

kfolds = KFold(n_splits=20, shuffle=True, random_state=42)



# Hyper parameters

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



# Regression models

ridge = RidgeCV(alphas=alphas_alt, cv=kfolds)

lasso = LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds)

elasticnet = ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio)                                

svr = SVR(C= 20, epsilon= 0.008, gamma=0.0003)



gbr = GradientBoostingRegressor(n_estimators=3000, 

                                learning_rate=0.05, 

                                max_depth=4, 

                                max_features='sqrt',

                                min_samples_leaf=15, 

                                min_samples_split=10, 

                                loss='huber', 

                                random_state =42)  



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

                                     reg_alpha=0.00006)



stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X_PCA):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)



# Scoring

score = cv_rmse(ridge)

print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(lasso)

print("Lasso: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(elasticnet)

print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )



score = cv_rmse(svr)

print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)



# Scoring

score = cv_rmse(ridge)

print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(lasso)

print("Lasso: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(elasticnet)

print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )



score = cv_rmse(svr)

print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()) )
print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))



print('elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)



print('Lasso')

lasso_model_full_data = lasso.fit(X, y)



print('Ridge')

ridge_model_full_data = ridge.fit(X, y)



print('Svr')

svr_model_full_data = svr.fit(X, y)



print('GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)



print('xgboost')

xgb_model_full_data = xgboost.fit(X, y)



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
def blend_models_predict(X):

    return ((0.01 * elastic_model_full_data.predict(X)) + \

            (0.01 * lasso_model_full_data.predict(X)) + \

            (0.01 * ridge_model_full_data.predict(X)) + \

            (0.15 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.2 * xgb_model_full_data.predict(X)) + \

            (0.22 * lgb_model_full_data.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))



print('RMSLE score:')

print(rmsle(y, blend_models_predict(X)))