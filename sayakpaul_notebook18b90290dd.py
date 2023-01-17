# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from sklearn.ensemble import StackingRegressor

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



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
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
test.head()
sample.head()
print("Training data shape :",train.shape)

print("Test data shape :",test.shape)
sns.set_style("darkgrid")

f, ax = plt.subplots(figsize=(8, 7))

#Check the new distribution 

sns.distplot(train['SalePrice'], color="r");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)

plt.show()
# Skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# log(1+x) transform

train["SalePrice"] = np.log1p(train["SalePrice"])
#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)

plt.show()
# Remove outliers

train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<12.5)].index, inplace=True)

train.reset_index(drop=True, inplace=True)
# Remove the Ids from train and test, as they are unique for each row and hence not useful for the model

train_ID = train['Id']

test_ID = test['Id']

train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
# Split features and labels

train_labels = train['SalePrice'].reset_index(drop=True)

train_features = train.drop(['SalePrice'], axis=1)

test_features = test



# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset

all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

all_features.shape
numeric = [col for col in train.columns if train[col].dtype in ['float64','int64','int32','float32','int16','float16']]

fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))

plt.subplots_adjust(right=2)

plt.subplots_adjust(top=2)

for i, feature in enumerate(list(train[numeric]), 1):

    if(feature=='MiscVal'):

        break

    plt.subplot(len(numeric), 3, i)

    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)

        

    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)

    plt.ylabel('SalePrice', size=15, labelpad=12.5)

    

    for j in range(2):

        plt.tick_params(axis='x', labelsize=12)

        plt.tick_params(axis='y', labelsize=12)

    

    plt.legend(loc='best', prop={'size': 10})
all_data_na = (all_features.isnull().sum() / len(all_features)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
# Visualize missing values

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

ax.xaxis.grid(False)

sns.despine(trim=True, left=True)
all_features['MSSubClass'].value_counts()
all_features.OverallCond.value_counts()
# convert them into strings 



all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)

all_features['YrSold'] = all_features['YrSold'].astype(str)

all_features['MoSold'] = all_features['MoSold'].astype(str)

all_features['OverallCond'] = all_features['OverallCond'].astype(str)
### Lets start imputing them ###

all_features["PoolQC"] = all_features["PoolQC"].fillna("None")

all_features["MiscFeature"] = all_features["MiscFeature"].fillna("None")

all_features["Alley"] = all_features["Alley"].fillna("None")

all_features["Fence"] = all_features["Fence"].fillna("None")

all_features["FireplaceQu"] = all_features["FireplaceQu"].fillna("None")



#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_features["LotFrontage"] = all_features.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_features[col] = all_features[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_features[col] = all_features[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_features[col] = all_features[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_features[col] = all_features[col].fillna('None')

all_features["MasVnrType"] = all_features["MasVnrType"].fillna("None")

all_features["MasVnrArea"] = all_features["MasVnrArea"].fillna(0)

all_features['MSZoning'] = all_features['MSZoning'].fillna(all_features['MSZoning'].mode()[0])

all_features = all_features.drop(['Utilities'], axis=1)

all_features["Functional"] = all_features["Functional"].fillna("Typ")

all_features['Electrical'] = all_features['Electrical'].fillna(all_features['Electrical'].mode()[0])

all_features['KitchenQual'] = all_features['KitchenQual'].fillna(all_features['KitchenQual'].mode()[0])

all_features['Exterior1st'] = all_features['Exterior1st'].fillna(all_features['Exterior1st'].mode()[0])

all_features['Exterior2nd'] = all_features['Exterior2nd'].fillna(all_features['Exterior2nd'].mode()[0])

all_features['SaleType'] = all_features['SaleType'].fillna(all_features['SaleType'].mode()[0])

all_features['MSSubClass'] = all_features['MSSubClass'].fillna("None")

#Check remaining missing values if any 

all_features_na = (all_features.isnull().sum() / len(all_features)) * 100

all_features_na = all_features_na.drop(all_features_na[all_features_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_features_na})

missing_data.head()
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_features[c].values)) 

    all_features[c] = lbl.transform(list(all_features[c].values))



# shape        

print('Shape all_features: {}'.format(all_features.shape))
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
# Fetch all numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numeric = []

for i in all_features.columns:

    if all_features[i].dtype in numeric_dtypes:

        numeric.append(i)
# Find skewed numerical features

skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))

skewness = pd.DataFrame({'Skew' :high_skew})

skew_features.head(10)
# Normalize skewed features

lam = 0.15

for i in skew_index:

    all_features[i] = boxcox1p(all_features[i],lam)
all_features = pd.get_dummies(all_features)

print(all_features.shape)
# Remove any duplicated column names

all_features = all_features.loc[:,~all_features.columns.duplicated()]
xtrain = all_features.iloc[:len(train_labels), :]

xtest = all_features.iloc[len(train_labels):, :]

print(xtrain.shape); print(train_labels.shape); print(xtest.shape)
pca = PCA(n_components=180)

xtrain =pca.fit_transform(xtrain)

xtest = pca.transform(xtest)
print(xtrain.shape); print(xtest.shape)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def rmse_cross_val(model, X=xtrain):

    rmse = np.sqrt(-cross_val_score(model, xtrain, train_labels, scoring="neg_mean_squared_error", cv=KFold(n_splits=12, random_state=42, shuffle=True)))

    return rmse
# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01,n_estimators=6000,max_depth=4,min_child_weight=0,gamma=0.6,

                       subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror',nthread=-1,scale_pos_weight=1,

                       seed=27,reg_alpha=0.00006,random_state=42)





# Light Gradient Boosting Regressor

#lightgbm = LGBMRegressor(objective='regression', num_leaves=6,learning_rate=0.01, n_estimators=7000,max_bin=200, 

 #                      bagging_fraction=0.8,bagging_freq=4, bagging_seed=8,feature_fraction=0.2,

  #                     feature_fraction_seed=8,min_sum_hessian_in_leaf = 11,verbose=-1,random_state=42)



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=6000,learning_rate=0.01,max_depth=4,max_features='sqrt',

                                min_samples_leaf=15,min_samples_split=10,loss='huber',random_state=42) 



# Random Forest Regressor

rf = RandomForestRegressor(n_estimators=1200,max_depth=15,min_samples_split=5,min_samples_leaf=5,

                          max_features=None,oob_score=True,random_state=42)



# Support Vector Regressor

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))



# Ridge Regressor

ridge_alphas = [1e-10, 1e-8, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=KFold(n_splits=12, random_state=42, shuffle=True)))



# Stack up all the models above, optimized using xgboost

#stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),meta_regressor=xgboost,random_state=42)



stack_reg = StackingRegressor(estimators=[('xgboost',xgboost),('svr',svr),('ridge',ridge),('gbr',gbr),('rf',rf)], final_estimator=xgboost)    
print('stack_reg')

stack_reg_model = stack_reg.fit(np.array(xtrain), np.array(train_labels))
#print('lightgbm')

#lgb_data = lightgbm.fit(xtrain, train_labels)
print('xgboost')

xgb_data = xgboost.fit(xtrain, train_labels)
print('Svr')

svr_data = svr.fit(xtrain, train_labels)
print('Ridge')

ridge_data = ridge.fit(xtrain, train_labels)
print('RandomForest')

rf_data = rf.fit(xtrain, train_labels)
print('GradientBoosting')

gbr_data = gbr.fit(xtrain, train_labels)
scores = {}



#score = rmse_cross_val(lightgbm)

#print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))

#scores['lgb'] = (score.mean(), score.std())
score = rmse_cross_val(svr)

print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['svr'] = (score.mean(), score.std())
score = rmse_cross_val(ridge)

print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['ridge'] = (score.mean(), score.std())
score = rmse_cross_val(rf)

print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['rf'] = (score.mean(), score.std())
score = rmse_cross_val(gbr)

print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['gbr'] = (score.mean(), score.std())
# Blend models in order to make the final predictions more robust to overfitting

def blended_predictions(xtrain):

    return ((0.2 * ridge_data.predict(xtrain)) + \

            (0.15 * svr_data.predict(xtrain)) + \

            (0.17 * gbr_data.predict(xtrain)) + \

            (0.1 * xgb_data.predict(xtrain)) + \

            (0.03 * rf_data.predict(xtrain)) + \

            (0.35 * stack_reg_model.predict(np.array(xtrain))))
# Get final precitions from the blended model

blended_score = rmsle(train_labels, blended_predictions(xtrain))

scores['blended'] = (blended_score, 0)

print('RMSE score on train data:')

print(blended_score)
# Plot the predictions for each model

sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(scores.values()):

    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')



plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)

plt.xlabel('Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)



plt.title('Scores of Models', size=20)

sample.head()
# Append predictions from blended models

sample.iloc[:,1] = np.floor(np.expm1(blended_predictions(xtest)))
# Fix outlier predictions

#q1 = sample['SalePrice'].quantile(0.0045)

#q2 = sample['SalePrice'].quantile(0.99)

#submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

#submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

sample.to_csv("my_submission.csv", index=False)