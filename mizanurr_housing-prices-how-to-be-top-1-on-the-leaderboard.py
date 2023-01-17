# Primary libraries

import numpy as np

import pandas as pd

import os



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import seaborn as sns

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt



#Stats

from scipy.stats import norm, skew

from scipy import stats

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



#Modelling

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV,LassoCV, Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



# Tensorflow

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split



# Others

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler,RobustScaler

from sklearn.decomposition import PCA



mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

# avoid warnings

import warnings

warnings.filterwarnings(action="ignore")
df_train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

df_test = pd.read_csv('../input/home-data-for-ml-course/test.csv')

print ("Data is loaded!")
df_train .head()
# See the name of the columns

df_train.columns
df_train.info()
df_train.describe()
%matplotlib inline

import matplotlib.pyplot as plt

df_train.hist(bins=40, figsize=(20,60))

#save_fig("attribute_histogram_plots")

plt.show()
#descriptive statistics summary

df_train['SalePrice'].describe()
#histogram

plt.figure(figsize=(10,5))

sns.distplot(df_train['SalePrice']);
print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#Scatter plot of SalePrice/GrLivArea

df_train.plot(kind="scatter", x="GrLivArea", y="SalePrice", alpha=1)
#Scatter plot of SalePrice/ TotalBsmtSF

df_train.plot(kind="scatter", x="TotalBsmtSF", y="SalePrice", alpha=1)
df_train.plot(kind="scatter", x="OverallQual", y="SalePrice", alpha=1)
# Remove outliers

df_train.drop(df_train[(df_train['GrLivArea']>4500) & (df_train['SalePrice']<300000)].index, inplace=True)

df_train.drop(df_train[(df_train['TotalBsmtSF']>6000) & (df_train['SalePrice']<300000)].index, inplace=True)

df_train.drop(df_train[(df_train['OverallQual']<4) & (df_train['SalePrice']>200000)].index, inplace=True)

df_train.reset_index(drop=True, inplace=True)
#box plot overallqual/saleprice

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=df_train["OverallQual"], y=df_train["SalePrice"])

fig.axis(ymin=0, ymax=800000);
#box plot overallqual/saleprice

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x=df_train["YearBuilt"], y=df_train["SalePrice"])

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corr_matrix = df_train.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)
attributes = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

scatter_matrix(df_train[attributes], figsize=(20,10))



plt.show()
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
# numpy fucntion log1p transforms it

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



#Check the new distribution 

sns.distplot(df_train['SalePrice'] , fit=norm);



#See the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
# Split features and labels

y = df_train['SalePrice'].reset_index(drop=True)

train_features = df_train.drop(['SalePrice'], axis=1)

test_features = df_test



# Combine train and test features

all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

all_features.shape
#first check the missing data

total = all_features.isnull().sum().sort_values(ascending=False)

percent = (all_features.isnull().sum()/all_features.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent*100], axis=1, keys=['Total', 'Percent'])

missing_data.head(35)
# Visualize missing values

missing_data=missing_data.head(35)

f, ax = plt.subplots(figsize=(16, 8))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.title('Percent missing data by feature', fontsize=15)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)
# PoolQC : data description says NA means "no pool"

all_features["PoolQC"] = all_features["PoolQC"].fillna("None")

# MiscFeature : data description says NA means "no misc feature"

all_features["MiscFeature"] =all_features["MiscFeature"].fillna("None")

# Alley : data description says NA means "no alley access"

all_features["Alley"] = all_features["Alley"].fillna("None")

# Fence : data description says NA means "no fence"

all_features["Fence"] = all_features["Fence"].fillna("None")

# FireplaceQu : data description says NA means "no fireplace"

all_features["FireplaceQu"] =all_features["FireplaceQu"].fillna("None")

# LotFrontage : filling missing values by the median LotFrontage of the neighborhood

all_features['LotFrontage'] = all_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# GarageType etc : data description says NA for garage features is "no garage"

for col in ( 'GarageCond', 'GarageQual','GarageType', 'GarageFinish'):

    all_features[col] = all_features[col].fillna('None')

for col in ('GarageYrBlt', 'GarageCars','GarageArea'):

    all_features[col] = all_features[col].fillna(0)

# BsmtQual etc : data description says NA for basement features is "no basement"

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

     all_features[col] =  all_features[col].fillna('None')       

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

     all_features[col] =  all_features[col].fillna(0)      

# MasVnrType : NA most likely means no veneer

all_features["MasVnrArea"] = all_features["MasVnrArea"].fillna(0)

all_features["MasVnrType"] = all_features["MasVnrType"].fillna("None")

# MSZoning : RL is the most common value

all_features['MSZoning'] = all_features['MSZoning'].fillna(all_features['MSZoning'].mode()[0])

#Functional : data description says Typ means typical

all_features["Functional"] = all_features["Functional"].fillna("Typ")

# Utilities : NA most likely means all public utilities

all_features["Utilities"] = all_features["Utilities"].fillna("AllPub")

# SBrkr is the most common value

all_features['Electrical'] = all_features['Electrical'].fillna("SBrkr")

#all_features['Electrical'] = all_features['Electrical'].fillna(all_features['Electrical'].mode()[0])

# KitchenQual : NA most likely means typical

all_features["KitchenQual"] = all_features["KitchenQual"].fillna("TA")

#Functional : data description says NA most likely means No building class          

all_features['MSSubClass'] = all_features['MSSubClass'].fillna("None")

# Replace few other missing values with their mode

all_features['Exterior1st'] = all_features['Exterior1st'].fillna(all_features['Exterior1st'].mode()[0])

all_features['Exterior2nd'] = all_features['Exterior2nd'].fillna(all_features['Exterior2nd'].mode()[0])

all_features['SaleType'] = all_features['SaleType'].fillna(all_features['SaleType'].mode()[0])
#convert them into strings

all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)

all_features['YrSold'] = all_features['YrSold'].astype(str)

all_features['MoSold'] = all_features['MoSold'].astype(str)
# Filter the skewed features

numeric_features = all_features.select_dtypes(include='number').columns

skew_features = all_features[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)



skewness = skew_features[skew_features > 0.5]

skew_index = skewness.index



skewness = pd.DataFrame({'Skew' :skewness})

skewness.head(10)
#fix skewed features

for i in skew_index:

    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
all_features["AllSF"] = all_features["GrLivArea"] + all_features["TotalBsmtSF"]

all_features["AllFlrsSF"] = all_features["1stFlrSF"] + all_features["2ndFlrSF"]

all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']

all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']

all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +

                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))

all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +

                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +

                              all_features['WoodDeckSF'])

all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']

all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')

all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +

                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])

all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)

all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1

all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1

all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1

all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1

all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1
def logs(columns):

    for col in columns:

        all_features[col+"_log"] = np.log(1.01+all_features[col])  



log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                 'TotalBsmtSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch','3SsnPorch','ScreenPorch','MiscVal','YearRemodAdd','TotalSF']



logs(log_features)
def squares(columns):

    for col in columns:

        all_features[col+"_sq"] =  all_features[col] * all_features[col]



squared_features = ['GarageArea_log','GarageCars_log','GrLivArea_log','YearRemodAdd', 'LotFrontage_log', 'TotalBsmtSF_log', '2ndFlrSF_log', 'GrLivArea_log' ]



squares(squared_features)
# Encode some categorical features as ordered numbers when there is information in the order

all_features = all_features.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                     )
#numerically encode categorical features

all_features = pd.get_dummies(all_features).reset_index(drop=True)
# let's see the shape one more time

all_features.shape
all_features.head()
# check is there any NAs in the dataset 

print("NAs for categorical features in the dataset : " + str(all_features.isnull().values.sum()))
X = all_features.iloc[:len(y), :]

X_test = all_features.iloc[len(y):, :]
X.shape, y.shape, X_test.shape
kf = KFold(n_splits=12, random_state=42, shuffle=True)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def rmse_cv(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
#We will also calculate the best alpha and ridge model in details

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10,15, 20,25, 30, 60,90])

ridge.fit(X, y)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4, alpha * 1.5, alpha * 1.6, alpha * 1.8], 

                cv = kf)

ridge.fit(X, y)

alpha = ridge.alpha_

print("Best alpha :", alpha)



alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4, alpha * 1.5, alpha * 1.6, alpha * 1.8]



ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas, cv=kf))
#We will also calculate the best alpha and lasso model in details

lasso = LassoCV(alphas = [0.0001, 0.0003,0.0005, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1], 

                max_iter = 50000, cv = kf)

lasso.fit(X, y)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4], 

                max_iter = 50000, cv = kf)
#We will also calculate the best alpha, ratio and the elasticNet model in details

elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7,0.75, 0.8, 0.85, 0.9, 0.95, 1],

                          alphas = [0.0001, 0.0003, 0.0005, 0.0006, 0.001, 0.003, 0.006, 

                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 5000, cv = kf)

elasticNet.fit(X, y)

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Try again for more precision with l1_ratio centered around " + str(ratio))

elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 0.8, 1, 3, 6], 

                          max_iter = 5000, cv = kf)

elasticNet.fit(X, y)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 

      " and alpha centered around " + str(alpha))

elasticNet = ElasticNetCV(l1_ratio = ratio,

                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 

                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 

                                    alpha * 1.35, alpha * 1.4], 

                          max_iter = 5000, cv = kf)

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))
KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)
gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=4060,gamma=0.0482,

                                     max_depth=4, min_child_weight=0,

                                     subsample=0.7,colsample_bytree=0.4603, 

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,reg_lambda=0.8571,

                                     reg_alpha=0.00006,random_state=42)
lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=6500,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=5, 

                       bagging_seed=9,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_data_in_leaf =6,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)
stackReg = StackingCVRegressor(regressors=(lightgbm,gbr, rf, ridge,svr,lasso),

                                meta_regressor=(xgboost),

                                use_features_in_secondary=True, 

                                random_state=42)
scores = {}



score = rmse_cv(ridge)

ridge_model= ridge.fit(X, y)

print("ridge : {:0.4f}".format(score.mean()))

scores['ridge'] = score.mean()
score = rmse_cv(lasso)

lasso_model= lasso.fit(X, y)

print("lasso : {:0.4f}".format(score.mean()))

scores['lasso'] = score.mean()
score = rmse_cv(svr)

svr_model= svr.fit(X, y)

print("svr: {:0.4f}".format(score.mean()))

scores['svr'] = score.mean()
score = rmse_cv(KRR)

KRR_model= KRR.fit(X, y)

print("KRR: {:0.4f}".format(score.mean()))

scores['KRR'] = score.mean()
score = rmse_cv(rf)

rf_model= rf.fit(X, y)

print("rf: {:0.4f}".format(score.mean()))

scores['rf'] = score.mean()
score = rmse_cv(gbr)

gbr_model= svr.fit(X, y)

print("gbr: {:0.4f}".format(score.mean()))

scores['gbr'] = score.mean()
score = rmse_cv(xgboost)

xgboost_model= xgboost.fit(X, y)

print("xgboost: {:0.4f}".format(score.mean()))

scores['xgboost'] = score.mean()
score = rmse_cv(lightgbm)

lgb_model= lightgbm.fit(X, y)

print("lightgbm: {:0.4f}".format(score.mean()))

scores['lgb'] = score.mean()
score = rmse_cv(stackReg)

stackReg_model= stackReg.fit(X, y)

print("stackReg: {:0.4f}".format(score.mean()))

scores['stackReg'] = score.mean()
model =Sequential()



model.add(Dense(358,activation='relu'))

model.add(Dropout(0.3))



model.add(Dense(180,activation='relu'))



model.add(Dense(90,activation='relu'))



model.add(Dense(45,activation='relu'))



model.add(Dense(20,activation='relu'))



model.add(Dense(10,activation='relu'))



model.add(Dense(4,activation='relu'))



model.add(Dense(1))



model.compile(optimizer='adam', loss='mse')
scale = StandardScaler()

X_train = scale.fit_transform(X)

X_val=scale.transform(X_test)
model.fit(X_train, y.values, epochs=50,batch_size=32)
loss_df=pd.DataFrame(model.history.history)

loss_df.plot()
#Check the rmse of the deep neural network model

rmse=mean_squared_error(model.predict(X_train), y)**0.5

rmse

scores['deepNeural'] = rmse
scores
#preeict Test Sales Price

yhat=np.expm1(model.predict(X_val))

yhat
#Also ignoring the deep neural network model as this can overfit the test data

def blendedModelPredictions(X,X_train):

    return ((0.15 * ridge_model.predict(X)) + \

            (0.2 * svr_model.predict(X)) + \

            (0.1 * gbr_model.predict(X)) + \

            (0.15 * xgboost_model.predict(X)) + \

            (0.1 * lgb_model.predict(X)) + \

            (0.3 * stackReg_model.predict(np.array(X))))
yhat11=blendedModelPredictions(X_test,X_val)

np.expm1(yhat11)
# blendedModelPrediction

blended_score = rmsle(y, blendedModelPredictions(X,X_train))

print("blended score: {:.4f}".format(blended_score))

scores['blended'] =  blended_score
pd.Series(scores).sort_values(ascending=True)
submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')

submission.iloc[:,1] = np.floor(np.expm1(blendedModelPredictions(X_test,X_val)))
submission.head()
submission.to_csv("submission_V1.csv", index=False)