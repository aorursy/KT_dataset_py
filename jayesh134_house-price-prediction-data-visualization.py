# All Necessary Imports

import os

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
# Files

print(os.listdir('../input/house-prices-advanced-regression-techniques/'))
# Locating our Data files

data_desc = '../input/house-prices-advanced-regression-techniques/data_description.txt'

train_path = '../input/house-prices-advanced-regression-techniques/train.csv'

test_path = '../input/house-prices-advanced-regression-techniques/test.csv'
# data description

data_desc = open(data_desc, mode='r')

for line in data_desc.readlines():

    print(line[:-1])
# Reading train and test data

train_data = pd.read_csv(train_path, index_col=0)

test_data = pd.read_csv(test_path, index_col=0)
# Having an eagle eye view of our data

train_data.head(3)
test_data.head(3)
# Columns and Rows in datasets



print('Train Data :\n\tRow Count : {r:}\n\tColumn Count : {c:}\n'.format(r=train_data.shape[0], c=train_data.shape[1]))

print('Test Data :\n\tRow Count : {r:}\n\tColumn Count : {c:}\n'.format(r=test_data.shape[0], c=test_data.shape[1]))
# Checking for only missing values in Train Data

print('Missing Values in Train Data:\n')

print(train_data.isna().sum()[train_data.isna().sum() != 0])
# Checking for only missing values in Test Data

print('Missing Values in Test Data :\n')

print(test_data.isna().sum()[test_data.isna().sum() != 0])
# We will just combine train and test data for feature engineering and later split it for modeling and testing.

# But first lets drop the extra col. from train data



data = pd.concat(objs=[train_data.drop('SalePrice', axis=1), test_data], axis=0)

print('Data :\n\tRow Count : {r:}\n\tColumn Count : {c:}'.format(r=data.shape[0], c=data.shape[1]))
# Lets count number of Numeric & Categorical Columns

count = 0

for col in data.select_dtypes(include=['integer', 'float']).columns:

    count += 1

print('Numeric Column Count : {:}'.format(count))



count = 0

for col in data.select_dtypes(include=['object']).columns:

    count += 1

print('Categorical Column Count : {:}'.format(count))
# Checking Correlation between Numeric values and Sale Price using heatmap

plt.figure(figsize=(25, 20))

sns.set(font_scale=1.4)

sns.heatmap(train_data.corr(), cmap='coolwarm', annot=True, fmt='.1f', linewidths=1.1, cbar=False)

plt.title('Correlation in Raw Training Data using Heatmap', fontsize=40)
# clustermap for train_data

sns.set(font_scale=1)

sns.clustermap(train_data.corr(), cmap='coolwarm', cbar=True, figsize=(20, 20), linewidths=0.05)

plt.title('Correlation in Raw Training Data using Clustermap', fontsize=20)
# Regession Plot : Checking Linear Regression Relation of every Numeric Column with Target 'SalePrice' with degree 1



fig, axes = plt.subplots(nrows=18, ncols=2, figsize=(28, 105))

fig.suptitle('Linear Relation between Numeric Feature & SalePrice',fontweight ="bold", fontsize=20)

sns.set(font_scale=1.8)



row=[x for x in range(0, 18)]

col=[0]

count = 1



for column in data.select_dtypes(include=['integer', 'float']).columns:

    # RegPlot

    sns.regplot(x=train_data[column][:1460], y=train_data['SalePrice'], ax=axes[row[0], col[0]],

               scatter_kws={"color": "black"}, line_kws={"color": "red"}, order=1)

    # Column

    if col[0] == 0:      

        col[0] = 1

    else :

        col[0] = 0

    # Row

    count += 1

    if count > 2:

        count = 1

        row.pop(0)

    fig.tight_layout()

    fig.subplots_adjust(top=0.97)
# train_data corr() bar plot

train_data.corr()['SalePrice'].sort_values(ascending=False).iplot(kind='bar', title='Correlation wrt SalePrice in Raw Training Data', color='red')
# Dropping columns

data.drop(labels=['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1, inplace=True)
# Filling numeric columns with their average values.

for col in data.select_dtypes(include=['integer', 'float']).columns:

    avg = data[col].mean()

    data[col].fillna(value=avg, inplace=True)
# Filling categorical columns with mode values.

for col in data.select_dtypes(include=['object']).columns:

    Mode = data[col].mode()[0]

    data[col].fillna(value=Mode, inplace=True)
# Confirming if anything is left

data.isna().sum()[data.isna().sum() != 0]
# Data Shape after Handling Missing Values

print('Data :\n\tRow Count : {r:}\n\tColumn Count : {c:}'.format(r=data.shape[0], c=data.shape[1]))



# Numeric & Categorical Columns left after Handling Missing Values

count = 0

for col in data.select_dtypes(include=['integer', 'float']).columns:

    count += 1

print('\tNumeric Column Count : {:}'.format(count))



count = 0

for col in data.select_dtypes(include=['object']).columns:

    count += 1

print('\tCategorical Column Count : {:}'.format(count))
# CountPlot : Checking number of occurence of every attribute in an individual category

# BoxPlot : Cheking behaviour of every attribute in an individual category wrt SalePrice



fig, axes = plt.subplots(nrows=38, ncols=2, figsize=(25, 215))

fig.suptitle('Relation between Categorical Features & SalePrice',fontweight ="bold", fontsize=20)

sns.set(font_scale=1.6)



row=[x for x in range(0, 38)]

col=[0]

count = 1



for column in data.select_dtypes(include=['object']).columns:

    for x in range(2):

        # Column

        if col[0] == 0:     # On 'Col 0' : CountPlot 

            figure1 = sns.countplot(x=data[column], ax=axes[row[0]][col[0]], 

                                    order=train_data.groupby(column)['SalePrice'].mean().sort_values().index)

            col[0] = 1

        elif col[0] == 1 :  # On 'Col 1' : BoxPlot

            figure2 = sns.boxplot(x=data[column][:1460], y=train_data['SalePrice'], ax=axes[row[0]][col[0]], 

                                  order=train_data.groupby(column)['SalePrice'].mean().sort_values().index)

            col[0] = 0

        # Row

        count += 1

        if count > 2:

            count = 1

            row.pop(0)

        fig.tight_layout()

        fig.subplots_adjust(top=0.978)
# Dropping Columns 

data.drop(labels=['Street', 'Utilities', 'Condition2', 'RoofMatl', 'BsmtFinType2',

      'Heating', 'Electrical', 'Functional', 'GarageCond', 'GarageQual'], axis=1, inplace=True)
# Mapping Categrical Columns

# Balancing Linearity 



# MSZoning

mszoning_map = {'FV':0, 'RL':1, 'RH':2, 'RM':2, 'C (all)':3}

data['MSZoning'] = data['MSZoning'].map(mszoning_map)



# LotShape

lotshape_map = {'IR2':0, 'IR3':1, 'IR1':1, 'Reg':2}

data['LotShape'] = data['LotShape'].map(lotshape_map)



# LotConfig

lotconfig_map = {'CulDSac':0, 'FR3':0, 'FR2':1, 'Corner':2, 'Inside':3}

data['LotConfig'] = data['LotConfig'].map(lotconfig_map)



# LandSlope

landslope_map = {'Mod':0, 'Sev':0, 'Gtl':1}

data['LandSlope'] = data['LandSlope'].map(landslope_map)



# Neighborhood

neighborhood_map = {'NoRidge':0, 'NridgHt': 0, 'StoneBr':1, 'Somerst':2, 'Timber':2, 'Veenker':2, 

                   'CollgCr':3, 'Crawfor':3, 'ClearCr':3 ,'Blmngtn':4, 'Gilbert':4, 'NWAmes':4,

                   'SawyerW':5, 'NAmes':6, 'Mitchel':6, 'NPkVill':6, 'Sawyer':7, 'Blueste':7, 'SWISU':7,

                   'BrkSide':8, 'Edwards':8, 'OldTown':8, 'MeadowV':9, 'IDOTRR':9, 'BrDale':9}

data['Neighborhood'] = data['Neighborhood'].map(neighborhood_map)



# Condition1

condition1_map = {'PosN':0, 'PosA':0, 'RRNn':0, 'RRNe':0, 'Norm':1, 'RRAn':2, 'RRAe':3, 'Feedr':3, 'Artery':4}

data['Condition1'] = data['Condition1'].map(condition1_map)



# BldgType

bldgtype_map = {'1Fam':0, 'TwnhsE':1, 'Twnhs':2, 'Duplex':3, '2fmCon':3}

data['BldgType'] = data['BldgType'].map(bldgtype_map)



# HouseStyle

housestyle_map = {'2Story':0, '2.5Fin':0, '1Story':1, 'SLvl':2, '2.5Unf':3, '1.5Fin':4, 'SFoyer':5, '1.5Unf':6}

data['HouseStyle'] = data['HouseStyle'].map(housestyle_map)



# Exterior1st

exterior1st_map = {'Stone':0, 'ImStucc':0, 'CemntBd':1, 'VinylSd':2, 'BrkFace':3, 'HdBoard':4, 'Plywood':4,

                  'Stucco':4, 'MetalSd':5, 'Wd Sdng':5, 'WdShing':5, 'AsbShng':6, 'CBlock':6, 'AsphShn':6, 'BrkComm':7}

data['Exterior1st'] = data['Exterior1st'].map(exterior1st_map)



# Exterior2nd

exterior2nd_map = {'Other':0, 'ImStucc':0, 'CmentBd':1, 'VinylSd':2, 'BrkFace':3, 'HdBoard':4, 'Plywood':4,

                  'Wd Shng':4, 'Stucco':5, 'Stone':5, 'MetalSd':6, 'Wd Sdng':6, 'AsphShn':7, 'Brk Cmn':7,

                  'AsbShng':8, 'CBlock':8}

data['Exterior2nd'] = data['Exterior2nd'].map(exterior2nd_map)



# BsmtFinType1

bsmtfintype1_map = {'GLQ':0, 'Unf':1, 'ALQ':2, 'LwQ':2, 'Rec':3, 'BLQ':3}

data['BsmtFinType1'] = data['BsmtFinType1'].map(bsmtfintype1_map)



# GarageType

garagetype_map = {'BuiltIn':0, 'Attchd':1, 'Basment':2, '2Types':2, 'Detchd':3, 'CarPort':4}

data['GarageType'] = data['GarageType'].map(garagetype_map)



# SaleType

saletype_map = {'New':0, 'Con':1, 'CWD':2, 'ConLI':2, 'WD':3, 'COD':4, 'ConLD':4, 'ConLw':4, 'Oth':4}

data['SaleType'] = data['SaleType'].map(saletype_map)



# SaleCondition

salecondition_map = {'Partial':0, 'Normal':1, 'Alloca':1, 'Abnorml':2, 'Family':2, 'AdjLand':3}

data['SaleCondition'] = data['SaleCondition'].map(salecondition_map)
# Applying Label Encoding 

le = LabelEncoder()

data[['LandContour', 'RoofStyle', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

      'BsmtQual', 'BsmtCond','BsmtExposure', 'HeatingQC', 'CentralAir', 'KitchenQual',

      'GarageFinish', 'PavedDrive']] = data[['LandContour', 'RoofStyle', 'MasVnrType', 'ExterQual', 'ExterCond',

       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC',

       'CentralAir', 'KitchenQual', 'GarageFinish', 'PavedDrive']].apply(lambda x : le.fit_transform(x))
# Checking if any categorical column has been missed

data.select_dtypes(include=['object']).columns
# Lets check for null values in data

data.isna().sum()[data.isna().sum() != 0]
# Dropping columns showing inter-corr with each other or zero corr with target value or each other

data.drop(labels=['GarageYrBlt', 'LandContour', 'MasVnrType', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath',

                  '3SsnPorch', 'MiscVal', 'MoSold', 'YrSold'], axis=1, inplace=True)
# Data Shape after Feature Engineering

print('Data :\n\tRow Count : {r:}\n\tColumn Count : {c:}'.format(r=data.shape[0], c=data.shape[1]))



# Numeric & Categorical Columns left after Handling Missing Values

count = 0

for col in data.select_dtypes(include=['integer', 'float']).columns:

    count += 1

print('\tNumeric Column Count : {:}'.format(count))



count = 0

for col in data.select_dtypes(include=['object']).columns:

    count += 1

print('\tObject Column Count : {:}'.format(count))
# Checking Correlation between Each Feature and Sale Price

data[:][:1460].corrwith(train_data['SalePrice']).sort_values(ascending=False).iplot(kind='bar', color='black',

                                                    title='Correlation of SalePrice with Every Other Feature')
# Seperating Train & Test Data



y = train_data['SalePrice']

train_data = data[:1460]

test_data = data[1460:]
data_2 = data[:1460]

data_2['SalePrice'] = y



# Checking Correlation between 54 features and Sale Price using heatmap

plt.figure(figsize=(25, 20))

sns.set(font_scale=0.9)

sns.heatmap(data_2.corr(), cmap='coolwarm', annot=True, fmt='.1f', linewidths=1.1, cbar=False)

plt.title('Correlation in Training Data using Heatmap', fontsize=40)
# Scaling 

scaler = StandardScaler()



train_data = pd.DataFrame(data=scaler.fit_transform(train_data), columns=train_data.columns)

test_data = pd.DataFrame(data=scaler.fit_transform(test_data), columns=test_data.columns)
# Algorithmns



# Linear Regression

lr = LinearRegression()



# Polynomial Regression degree 2

poly_feature_2 = PolynomialFeatures(degree=2)

poly_data_2 = poly_feature_2.fit_transform(train_data)

poly_reg_2 = LinearRegression()



# Polynomial Regression degree 3

poly_feature_3 = PolynomialFeatures(degree=3)

poly_data_3 = poly_feature_3.fit_transform(train_data)

poly_reg_3 = LinearRegression()



# Ridge Regression

alphas_r = [0.005, 0.001, 0.01, 0.1, 1, 3, 5, 8, 10, 12, 15, 18]

ridge_reg = RidgeCV(alphas=alphas_r, cv=10, scoring='neg_mean_squared_error')



# Lasso Regression

lasso_reg = LassoCV(cv=10)



# ElasticNetCV Regression

elasticnet_reg = ElasticNetCV(cv=10)



# Support Vector Regressor

sv_reg = SVR()



# Random Forest Regressor

rfr_params = {'n_estimators':[100, 150, 200, 250, 300, 350, 400]}

rfr_reg = GridSearchCV(estimator=RandomForestRegressor(), param_grid=rfr_params, scoring='neg_mean_squared_error', cv=10)



# AdaBoost Regressor

adaboost_params = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 420, 480, 550]}

adaboost_reg = GridSearchCV(estimator=AdaBoostRegressor(), param_grid=adaboost_params, scoring='neg_mean_squared_error', cv=10)



# GradientBoosting Regressor

gbm_params = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 420, 480, 550]}

gbm_reg = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=gbm_params, scoring='neg_mean_squared_error', cv=10)



# Light GradientBoosting Regressor

lgbm_params = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 420, 480, 550]}

lgbm_reg = GridSearchCV(estimator=LGBMRegressor(), param_grid=lgbm_params, scoring='neg_mean_squared_error', cv=10)



# XGBoost Regressor

xgbm_params = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 420, 480, 550]}

xgbm_reg = GridSearchCV(estimator=XGBRegressor(), param_grid=xgbm_params, scoring='neg_mean_squared_error', cv=10)
# Non Ensemble models

models_regular = {'RidgeCV':ridge_reg, 'LassoCV':lasso_reg, 'ElasticNetCV':elasticnet_reg, 

                   'Linear Regression':lr, 'Polynomial Regression(degree=2)': poly_reg_2,

                   'Polynomial Regression(degree=3)': poly_reg_3, 'Support Vector Regressor':sv_reg}



# Ensemble models 

models_ensemble = {'Random Forest Regressor':rfr_reg,'Adaboost Regressor':adaboost_reg,'GradientBoosting Regressor':gbm_reg,

          'Light GradientBoosting Regressor':lgbm_reg, 'XGBoost Regressor':xgbm_reg}



# Dictionary to keep model performance records i.e cross_validate scores 

Record = dict()
# Collecting score : 'mean square errors' for non-ensemble models with cross_validate

# u can directly get 'mse' after fitting model ----> model.mse_path_ (i.e mean square error for the test set on each fold, varying alpha)

# but i want 'mse' for train as well so will use cross_vaidate instead of cross_val_score



for model_name, model in models_regular.items():

    if  model_name == 'Polynomial Regression(degree=2)' :     # Polynomial regression takes in poly_data with degree 2

        model.fit(X=poly_data_2, y=y)

        score = cross_validate(model, poly_data_2, y, scoring='neg_root_mean_squared_error', cv=10, verbose=1, return_train_score=True)

    elif model_name == 'Polynomial Regression(degree=3)' :    # Polynomial regression takes in poly_data with degree 3

        model.fit(X=poly_data_3, y=y)

        score = cross_validate(model, poly_data_3, y, scoring='neg_root_mean_squared_error', cv=10, verbose=1, return_train_score=True)

    else :   

        model.fit(X=train_data, y=y)

        score = cross_validate(model, train_data, y, scoring='neg_root_mean_squared_error', cv=10, verbose=1, return_train_score=True)

    

    score['test RMSE'] = -score['test_score'].mean()

    score['train RMSE'] = -score['train_score'].mean()

    score['fit time'] = score['fit_time'].mean()

    score['score time'] = score['score_time'].mean()

    

    Record[model_name] = score
# Collecting score : 'mean square errors' for ensemble models with cross_validate

for model_name, model in models_ensemble.items():

    model.fit(X=train_data, y=y)

    score = cross_validate(model.best_estimator_, train_data, y, scoring='neg_root_mean_squared_error', cv=10, verbose=1, return_train_score=True)

    

    score['test RMSE'] = -score['test_score'].mean()

    score['train RMSE'] = -score['train_score'].mean()

    score['fit time'] = score['fit_time'].mean()

    score['score time'] = score['score_time'].mean()

    

    Record[model_name] = score
# Creating our Model Performance Report

efficiency_report = pd.DataFrame(Record).T.drop(['test_score', 'train_score', 'fit_time', 'score_time'], axis=1).sort_values(by='test RMSE')

efficiency_report
# Creating our Model Performance Report considering 54 features, dropping corr==0

efficiency_report = pd.DataFrame(Record).T.drop(['test_score', 'train_score', 'fit_time', 'score_time'], axis=1).sort_values(by='test RMSE')

efficiency_report
# stacking model

stacking_reg = StackingRegressor(estimators=[('gbr', gbm_reg.best_estimator_), ('xgbr', xgbm_reg.best_estimator_),

                                             ('lgbr', lgbm_reg.best_estimator_), ('rfr', rfr_reg.best_estimator_),

                                             ('ridgeCV', ridge_reg), ('lr', lr), ('lassoCV', lasso_reg), 

                                             ('adaboost_reg', adaboost_reg.best_estimator_)], 

                                 final_estimator=RandomForestRegressor(n_estimators=180), cv=10)



# cross_validate stacked model

score = cross_validate(stacking_reg, train_data, y, scoring='neg_root_mean_squared_error', cv=10, verbose=1, return_train_score=True)

score['test RMSE'] = -score['test_score'].mean()

score['train RMSE'] = -score['train_score'].mean()

score['fit time'] = score['fit_time'].mean()

score['score time'] = score['score_time'].mean()



# Taking into account model performance report

Record['Stacked Regression'] = score
# Stacked Model Training

stacking_reg.fit(X=train_data, y=y)
# Creating our Model Performance Report

efficiency_report = pd.DataFrame(Record).T.drop(['test_score', 'train_score', 'fit_time', 'score_time'], axis=1).sort_values(by='test RMSE')

efficiency_report
# Prediction using Stacked Models

final_pred = stacking_reg.predict(test_data)
# Submission

submission = pd.DataFrame()



submission['Id'] = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')['Id']

submission['SalePrice'] = final_pred



submission.to_csv('House Price Prediction.csv',index=False)

pd.read_csv('House Price Prediction.csv', index_col=0).head(8)