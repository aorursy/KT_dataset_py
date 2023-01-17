# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
# Import section



import pandas as pd

import numpy as np



import seaborn as sns

import types

import pandas as pd

from botocore.client import Config

#import ibm_boto3



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



# Misc

import sklearn.metrics as metrics

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

from datetime import datetime



pd.set_option('display.max_columns', None)



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000
trainDf = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
trainDf.head(10)
testDf = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
testDf.head()
# First let us look at the Target and plot it visually

import matplotlib.pyplot as plt



sns.set_style ("white")

sns.set_color_codes (palette = 'deep')

f, ax = plt.subplots (figsize=(8, 7))



sns.distplot (trainDf['SalePrice'], color='b')

ax.xaxis.grid =False

ax.set (ylabel='Frequency' )

ax.set (ylabel='SalePrice' )

ax.set (title = 'SalePrice Distribution')

sns.despine(trim=True, left=True)

plt.show()
# Skewness and Kurtosis



print ("Skewness of Data : %.2f" % trainDf['SalePrice'].skew())

print ("Kurtosis of Data : %.2f" % trainDf['SalePrice'].kurt())
pd.set_option('display.float_format', lambda x: '%.2f' %x)

trainDf.describe()
trainDf.describe(include = ['object'], exclude = ['int', 'float'])
fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(10, 8))

fig.tight_layout(pad=6.0)

sns.boxplot(x='OverallQual', y='SalePrice', data=trainDf, orient='v', ax=axes[0])

sns.boxplot(x='OverallCond', y='SalePrice', data=trainDf, orient='v', ax=axes[1])
NumericColumns = trainDf.select_dtypes([np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]).columns

NumericColumns

FeaturePlot = NumericColumns.drop([ 'SalePrice'])
ColumnDisplay = 2

fig, axex = plt.subplots( ncols=ColumnDisplay, nrows=0, figsize=(12,120) )



plt.subplots_adjust (top=2, right=2)

sns.color_palette ('husl', 8)

for i, feature in enumerate (list(trainDf[FeaturePlot]), 1) :

    plt.subplot (len(list(FeaturePlot)), ColumnDisplay, i )

    sns.scatterplot (x=feature, y='SalePrice', data=trainDf, hue='SalePrice', palette='Blues')
CategoricColumns = trainDf.select_dtypes([np.object]).columns

#CategoricColumns
DisplayColumns=2

fig, axes = plt.subplots (ncols=DisplayColumns, nrows=0, figsize=(12, 120))

plt.subplots_adjust (top=2, right=2)

sns.color_palette('RdGy', 10)



for i, feature in enumerate (list(trainDf[CategoricColumns]), 1) :

    plt.subplot (len(list(CategoricColumns)), ColumnDisplay, i )

    sns.boxplot (x=feature, y='SalePrice', data=trainDf, orient='v')
trainDf['TranSalePrice'] = np.log1p(trainDf['SalePrice'])

trainDf[['Id', 'SalePrice', 'TranSalePrice']].head()
fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(8, 6))



fig.tight_layout(pad=4.0)



#Set the generic properties of Seaborn

sns.set_style("white")

sns.set_color_codes (palette = 'deep')

sns.despine(trim=True, left=True)



# The first distribution plot is for the original SalePrice data

sns.distplot(trainDf['SalePrice'], color="b", ax=axes[0]);

#ax.grid(False)

axes[0].set(ylabel="Frequency")

axes[0].set(xlabel="SalePrice")

#axes[0].xticks(rotation=90)

axes[0].set(title="SalePrice distribution-Original")



# The Second distribution plot is for the original SalePrice data

sns.distplot(trainDf['TranSalePrice'], fit=norm, color="g", ax=axes[1]);

(mu, sigma) = norm.fit(trainDf['TranSalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

#ax.xaxis.grid(False)

axes[1].set(ylabel="Frequency")

axes[1].set(xlabel="SalePrice")

axes[1].set(title="SalePrice distribution-Transform")



plt.show()
# Remove outliers



trainDf.drop(trainDf[(trainDf['GrLivArea']>4000) & (trainDf['SalePrice']<300000)].index, inplace=True)

trainDf.reset_index(drop=True, inplace=True)
trainEnd = trainDf.shape[0] #retain the count for segregation in future

CombineDf = pd.concat ([trainDf, testDf], sort=True).reset_index(drop=True)

#trainEnd
CombineDf.head()
CombineDf.drop(['Id', 'SalePrice', 'TranSalePrice'], axis=1, inplace=True)
# Let us check the extent of values that are NaN (missing Values)



# Create a subroutine to list down the NaN % in a tabular form. This subroutine will be invoked multiple times.

def ListEmptiness (df) :

    CombineNaN = (df.isnull().sum()/df.shape[0]) * 100 # Get the % of the Attributes that have Null value

    CombineNaN = CombineNaN[CombineNaN !=0].sort_values(ascending=False)

    nanData = pd.DataFrame({'Nan Ratio': CombineNaN})

    return nanData



Emptyness = ListEmptiness (CombineDf)

#Emptyness
#Visualize the Missing Attributes

f,ax = plt.subplots (figsize=(10,8))

sns.barplot (y='Nan Ratio', x=Emptyness.index, data=Emptyness)

plt.xticks(rotation=90);

plt.ylabel('Percentage of Missing data in the Feature')

plt.xlabel('Features')

plt.title('Missng data by Feature');
CombineDf['PoolQC'].value_counts(dropna=False).to_frame()
CombineDf['PoolArea'].describe()
CombineDf['PoolQC'] = CombineDf['PoolQC'].fillna("None")
CombineDf['MiscFeature'].value_counts(dropna=False).to_frame()
CombineDf['MiscFeature'] = CombineDf['MiscFeature'].fillna("None")
CombineDf['Alley'] = CombineDf['Alley'].fillna("None")
CombineDf['Fence'] = CombineDf['Fence'].fillna ("None")
CombineDf['FireplaceQu'] = CombineDf['FireplaceQu'].fillna ("None")
CombineDf['LotFrontage'] = CombineDf.groupby('Neighborhood')['LotFrontage'].transform (lambda x:x.fillna(x.median()))
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

        CombineDf[col] = CombineDf[col].fillna("None")

# Replacing the missing values with 0, since no garage = no cars in garage

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

        CombineDf[col] = CombineDf[col].fillna(0)
# NaN values for these categorical basement features, means there's no basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    CombineDf[col] = CombineDf[col].fillna('None')
# Create a Multi-culumn display to visualize same category columns

from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

        html_str+="<td>&nbsp&nbsp&nbsp</td>"

    #print(html_str)

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
display_side_by_side (CombineDf['Electrical'].value_counts(dropna=False).to_frame(), \

                      CombineDf['Functional'].value_counts(dropna=False).to_frame(), \

                      CombineDf['Utilities'].value_counts(dropna=False).to_frame(), \

                      CombineDf['SaleType'].value_counts(dropna=False).to_frame(), \

                      CombineDf['KitchenQual'].value_counts(dropna=False).to_frame()

                     )
ColumnList = {'Electrical', 'Functional', 'Utilities', 'SaleType', 'KitchenQual' }

for col in ColumnList :

    #print (CombineDf[col].mode()[0])

    CombineDf[col] = CombineDf[col].fillna (CombineDf[col].mode()[0] )
display_side_by_side (

                      CombineDf['Exterior1st'].value_counts(dropna=False).to_frame(), \

                      CombineDf['Exterior2nd'].value_counts(dropna=False).to_frame() \

                     )
ColumnList = { 'Exterior1st', 'Exterior2nd' }

for col in ColumnList :

    #print (CombineDf[col].mode()[0])

    CombineDf[col] = CombineDf[col].fillna (CombineDf[col].mode()[0] )
CombineDf['BsmtFinSF1' ] = CombineDf['BsmtFinSF1'].fillna(0)



CombineDf['BsmtFinSF2' ] = CombineDf['BsmtFinSF2'].fillna(0)



CombineDf['BsmtFullBath' ] = CombineDf['BsmtFullBath'].fillna(0)

CombineDf['BsmtHalfBath' ] = CombineDf['BsmtHalfBath'].fillna(0)

CombineDf['BsmtUnfSF' ] = CombineDf['BsmtUnfSF'].fillna(0)

CombineDf['TotalBsmtSF' ] = CombineDf['TotalBsmtSF'].fillna(0)

display_side_by_side (CombineDf['MasVnrType'].value_counts(dropna=False).to_frame(), \

                      #CombineDf['MasVnrArea'].value_counts(dropna=False).to_frame(), \

                      CombineDf['MSZoning'].value_counts(dropna=False).to_frame() )
CombineDf['MasVnrType'] = CombineDf['MasVnrType'].fillna(CombineDf['MasVnrType'].mode()[0])



CombineDf['MasVnrArea'] = CombineDf['MasVnrArea'].fillna(0)
CombineDf['MSZoning'] = CombineDf.groupby('Neighborhood')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# Let's make sure we handled all the missing values



Emptyness = ListEmptiness (CombineDf)

Emptyness
NumericColumns = CombineDf.select_dtypes([np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]).columns

#NumericColumns
# Create box plots for all numeric features

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

ax.set_xscale("log")

ax = sns.boxplot(data=CombineDf[NumericColumns] , orient="h", palette="RdGy")

ax.xaxis.grid(False)

ax.set(ylabel="Feature names")

ax.set(xlabel="Numeric values")

ax.set(title="Numeric Distribution of Features")

sns.despine(trim=True, left=True)
SkewFeatures = CombineDf[NumericColumns].apply(lambda x: skew(x)).sort_values (ascending=False)



HighSkews = SkewFeatures[SkewFeatures > 0.5]

SkewIndex = HighSkews.index



print('There are {} Numerical Features with High Skew Values'.format(SkewIndex.shape[0]))
for i in SkewIndex :

    CombineDf[i] = boxcox1p( CombineDf[i], boxcox_normmax(CombineDf[i] +1) )
SkewFeatures = CombineDf[NumericColumns].apply(lambda x: skew(x)).sort_values (ascending=False)



HighSkews = SkewFeatures[SkewFeatures > 0.5]

SkewIndex = HighSkews.index



print('There are {} Numerical Features with High Skew Values'.format(SkewIndex.shape[0]))
CombineDf['YearsSinceRemodel'] = CombineDf['YrSold'].astype(int) - CombineDf['YearRemodAdd'].astype(int)

CombineDf['Total_Home_Quality'] = CombineDf['OverallQual'] + CombineDf['OverallCond']

CombineDf['TotalSF'] = CombineDf['TotalBsmtSF'] + CombineDf['1stFlrSF'] + CombineDf['2ndFlrSF']

CombineDf['Total_Bathrooms'] = (CombineDf['FullBath'] + (0.5 * CombineDf['HalfBath']) +\

                               CombineDf['BsmtFullBath'] + (0.5 * CombineDf['BsmtHalfBath']))
CombineDf = CombineDf.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
train= pd.concat([CombineDf.iloc[:trainEnd, :], trainDf['TranSalePrice']], axis=1)

train_corr = train.corr()
mask = np.zeros_like(train_corr)

mask[np.triu_indices_from(mask)] = True



cmap = sns.diverging_palette (180, 30, as_cmap=True)



with sns.axes_style("white"):

     fig, ax = plt.subplots(figsize=(13,11))

     sns.heatmap(train_corr, vmax=.8, mask=mask, cmap=cmap, cbar_kws={'shrink':.5}, linewidth=.05);
DummyCombineDf = pd.get_dummies(CombineDf)
print ('Shape of the Dataset (Train+ Test) Rows:{}, Columns:{}'.format(DummyCombineDf.shape[0], DummyCombineDf.shape[1]))
X_Train = DummyCombineDf.iloc[ : trainEnd, :]

X_Test  = DummyCombineDf.iloc[trainEnd :, :] 



#y_train = trainDf[['TranSalePrice']]

y_train = trainDf['TranSalePrice'].reset_index(drop=True)
y_train.head(5)
#Common Params

Kf = KFold(n_splits=4, random_state=42, shuffle=True) # Number of K-Folds



# Light Gradient Boosting Regressor

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=7000,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

#                       objective='reg:linear',

                       objective='reg:squarederror',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42)



# Ridge Regressor

#ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]

ridge_alphas = [ 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=Kf))



# Support Vector Regressor

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)  



# Random Forest Regressor

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)



# Stack up all the models above, optimized using xgboost

stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
def cv_rmse(model, TrainFeature, TrainTarget):

    rmse = np.sqrt(-cross_val_score(model, TrainFeature, TrainTarget, scoring="neg_mean_squared_error", cv=Kf))

    return (rmse)


Scores = {}





print("About to start Scoring for First Set Algorithms Time:%s" % datetime.now())

for clf, label in zip([ridge, svr, rf], [ 'Ridge', 'SVM', 'Random Forest']):

    score = cv_rmse(clf, X_Train, y_train)

    print("Neg. MSE Score: %0.4f (+/- %0.4f) [%s] Time::%s" % ( score.mean(), score.std(), label, datetime.now()))

    Scores[label] = (score.mean(), score.std())

    

print("About to start Scoring for Second Set Algorithms Time:%s" % datetime.now())



for clf, label in zip([ xgboost, gbr, lightgbm], [ 'xgBoost', 'GradientBooster', 'lightGBM']):

    score = cv_rmse(clf, X_Train, y_train)

    print("Neg. MSE Score: %0.4f (+/- %0.4f) [%s] Time::%s" % ( score.mean(), score.std(), label, datetime.now()))

    Scores[label] = (score.mean(), score.std())
Scores
print (X_Train.shape, y_train.shape)
print('stack_gen Start Time:%s' %  datetime.now())

stack_gen_model = stack_gen.fit(np.array(X_Train), np.array(y_train) )

print('stack_gen End   Time:%s' % datetime.now())
print('lightgbm Start Time:%s' %  datetime.now())

lightgbm_gen_model = lightgbm.fit(X_Train, y_train )

print('lightgbm End   Time:%s' % datetime.now())
print('xgBoost Start Time:%s' %  datetime.now())

xgb_gen_model = xgboost.fit(X_Train, y_train )

print('xgBoost End   Time:%s' % datetime.now())
print('SVR Start Time:%s' %  datetime.now())

svr_gen_model = svr.fit(X_Train, y_train )

print('SVR End   Time:%s' % datetime.now())
print('Ridge Start Time:%s' %  datetime.now())

ridge_gen_model = ridge.fit(X_Train, y_train )

print('Ridge End   Time:%s' % datetime.now())
print('Random Forest Start Time:%s' %  datetime.now())

rf_gen_model = rf.fit(X_Train, y_train )

print('Random Forest End   Time:%s' % datetime.now())
print('GradientBoosting  Start Time:%s' %  datetime.now())

gbr_gen_model = gbr.fit(X_Train, y_train )

print('GradientBoosting  End   Time:%s' % datetime.now())
# Blend models in order to make the final predictions more robust to overfitting

def blended_predictions(XBlend):

    #print (XBlend.shape)

    #print(XBlend.columns)

    return (   (0.1  * ridge_gen_model.predict(XBlend))  \

             + (0.2  * svr_gen_model.predict(XBlend))  \

             + (0.1  * gbr_gen_model.predict(XBlend)) \

             + (0.1  * xgb_gen_model.predict(XBlend))  \

             + (0.1  * lightgbm_gen_model.predict(XBlend))  \

             + (0.05 * rf_gen_model.predict(XBlend))  \

             + (0.35 * stack_gen_model.predict(np.array(XBlend)))

            )

# Get final precitions from the blended model

print('Blended   Start Time:%s' %  datetime.now())

Blended_Yhat = blended_predictions(X_Train)

print('Blended   End Time:%s' %  datetime.now())



Blended_Yhat
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
blended_score = rmsle(y_train, blended_predictions(X_Train))



Scores['blended'] = (blended_score, 0)

print('RMSLE score on train data:')

print(blended_score)
# Plot the predictions for each model

sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



ax = sns.pointplot(x=list(Scores.keys()), y=[score for score, _ in Scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(Scores.values()):

    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')



plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)

plt.xlabel('Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)



plt.title('Scores of Models', size=20)



plt.show()
#Apply Prediction to the Test dataset using the best Regression Algorithm (in this case Ridge)

testDf['TranSalePrice'] = blended_predictions(X_Test)
#Get the SalePrice via the inverse Normalization tranformation

testDf['SalePrice'] = np.floor(np.expm1(testDf['TranSalePrice']))
testDf[['Id', 'TranSalePrice', 'SalePrice']].head(10).style.format({'SalePrice': "{:,.0f}"})
my_submission = pd.DataFrame({'Id': testDf.Id, 'SalePrice': testDf.SalePrice})

# Use any filename. I choose submission here



my_submission.to_csv('submission.csv', index=False)
