# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

#importing supportive libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #Data Visualization
import matplotlib.pyplot as plt #Data Visualization
from scipy.stats import skew #Function to Determine skewness associated with variables in the data
from scipy.stats.stats import pearsonr #To find Correlation coefficient
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
#Models for Prediction problem
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet,Ridge, BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
# import xgboost as xgb
from sklearn.pipeline import make_pipeline
#functions to support data splitting, Data Transformation and evaluation metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from matplotlib import rcParams
import os
# print(os.listdir("./input"))
#training data
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

#test data
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# DATA PREPARATION
#making a copy of both train and test set for future reference and reduce redundancy in loading the data again
train_copy = train.copy()
test_copy = test.copy()

#Concatenate train and test
fulldata = pd.concat([train, test])
fulldata = fulldata.reset_index(drop = True)
missing_data = train.isnull().apply(sum).sort_values(ascending = False)
missing_col_name = missing_data[missing_data > 0]
# print(missing_col_name)
# print("There are {} variables with missing values".format(len(missing_col_name)))

#Fetching the names of the attributes that has missing values
missing_col_name = missing_data[missing_data > 0]
corr_matrix_missing = train[missing_col_name.index]
corr_matrix_missing["SalePrice"] = train["SalePrice"]

missing_corr_values = corr_matrix_missing.corr()

#checking numerical variables correlation as well as finding out how many categorical variables have missing value.
# print(missing_corr_values["SalePrice"].sort_values(ascending = False))

fulldata_missing = fulldata.isnull().sum().sort_values(ascending = False)
fulldata_missing_colname = fulldata_missing[fulldata_missing > 0]
# print(fulldata_missing_colname)
#features that has same conventions as Alley Access - filled with None. Note - I am giving "None" instead of zero here because they are Qualitaive varaibles.
#The categories are class labels in text. We will later convert some of these variables using LabelEncoder.
NoneFill = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageType", "GarageFinish", "GarageQual", "GarageCond","BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2","MasVnrType"]

for item in NoneFill:
    fulldata[item] = fulldata[item].fillna("None")


#These categorical Labels already are in numeric categorical form and hence do not require encoding, however, conversion to dummy variables can be done
ZeroFill = ['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea']

for item in ZeroFill:
    fulldata[item] = fulldata[item].fillna(0)


fulldata["LotFrontage"] = fulldata.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

ModeFill = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','Utilities']

for item in ModeFill:
    fulldata[item] = fulldata[item].fillna(fulldata[item].mode()[0])

fulldata["Functional"] = fulldata["Functional"].fillna("Typ")
#Dropping the SalePirce variable from fulldata
fulldata.drop("SalePrice", axis = 1, inplace = True)

#Checking if Missing values still persist in our data
# print(fulldata.isnull().any().any())
#Outliers

f, ax = plt.subplots()
ax.scatter(x = train["GrLivArea"],y = train["SalePrice"])
ax.set_xlabel("Ground Living Area")
ax.set_ylabel("Sale Price")
f,ax = plt.subplots()
ax.scatter(x = train["TotalBsmtSF"], y = train["SalePrice"])
ax.set_xlabel("Total basement Square feet")
ax.set_ylabel("Sale Price")

TotalBsmtSF_row = train.loc[(train["TotalBsmtSF"] > 6000) & (train["SalePrice"] < 200000)]
GrLivArea_row = train.loc[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)]
# print(TotalBsmtSF_row)
# print(GrLivArea_row)

#Remving the outliers from both the dataset - kindly note to follow the same order while removing the oulier.
#1st remove the outlier from fulldata dataset then remove from train dataset
#Because once you remove the outlier from train data, you wont get the index to remove it from fulldata.
fulldata = fulldata.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)
fulldata = fulldata.reset_index(drop = True)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)
train = train.reset_index(drop = True)

train.columns.to_series().groupby(train.dtypes).groups

#MSSubClass=The building class
fulldata['MSSubClass'] = fulldata['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
fulldata['OverallCond'] = fulldata['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
fulldata['YrSold'] = fulldata['YrSold'].astype(str)
fulldata['MoSold'] = fulldata['MoSold'].astype(str)
fulldata['YearBuilt'] = fulldata['YearBuilt'].astype(str)
fulldata['YearRemodAdd'] = fulldata["YearRemodAdd"].astype(str)
# Adding a new total feature
fulldata['TotalSF'] = fulldata['TotalBsmtSF'] + fulldata['1stFlrSF'] + fulldata['2ndFlrSF']

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold','YearBuilt', "YearRemodAdd")
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(fulldata[c].values))
    fulldata[c] = lbl.transform(list(fulldata[c].values))
    
# shape
print('Shape all_data: {}'.format(fulldata.shape))
# DATA TRANSFORMATIONS

int_features = fulldata.dtypes[fulldata.dtypes == "int64"].index
float_features = fulldata.dtypes[fulldata.dtypes == "float64"].index

# Check the skew of all numerical features
skewed_int_feats = fulldata[int_features].apply(lambda x: skew(x.dropna()))
skewed_float_feats = fulldata[float_features].apply(lambda x: skew(x.dropna()))

skewed_features = pd.concat([skewed_int_feats,skewed_float_feats])

print("\nSkewness in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_features})
skewness = skewness.sort_values('Skew', ascending = False)
skewness.head(15)
skewness = skewness[abs(skewness) > 0.5]
print("There are {} skewed numerical features to log(x+1) transform".format(skewness.shape[0]))
skewed_features = skewness.index
for features in skewed_features:
    fulldata[features] = np.log1p(fulldata[features])

    fulldata = pd.get_dummies(fulldata)
print(fulldata.shape)

sns.distplot(train['SalePrice'])

train["SalePrice"] = np.log1p(train["SalePrice"])

y_train = train.SalePrice.values

x_train = fulldata[:train.shape[0]]

print("x_train shape:{}".format(x_train.shape))
print("y_train shape:{}".format(y_train.shape))
# plt.show()


models = [['DecisionTree :',DecisionTreeRegressor()],
           ['RandomForest :',RandomForestRegressor()],
           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
           ['KernelRidge:',KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)],
           ['LassoLarsIC :',LassoLarsIC()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)],
           ['Lasso: ', Lasso(alpha =0.0005, random_state=1)],
           ['Ridge: ', Ridge()],
           ['BayesianRidge: ', BayesianRidge()],
           ['ElasticNet: ', ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)]]


print("Score of Models...")

print(make_pipeline(StandardScaler(),DecisionTreeRegressor()))
# for name,model in models:
#     ModelTemp = make_pipeline(StandardScaler(),model)
#     rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = 10))
#     print("Average {} cross validation score: ".format(name), np.mean(rmse))
DecisionTreeRegModel = DecisionTreeRegressor()
DecisionTreeRegModel.fit(x_train.values, y_train)
pred = DecisionTreeRegModel.predict(x_train)
# print(pred)
# print(x_train)
# print(x_train.values)
print(x_train)
print(fulldata.shape)

dd = pd.concat([train_copy, test_copy])
import os
import warnings
warnings.simplefilter(action = 'ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import pylab 
from scipy import stats
from scipy.stats import skew, norm, probplot, boxcox
#from scipy.special import boxcox1p
#from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFECV, SelectFromModel
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA, KernelPCA
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, Lasso, LassoLarsIC, ElasticNet, ElasticNetCV
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

train.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)
test.rename(columns={'3SsnPorch':'TSsnPorch'}, inplace=True)

test['SalePrice'] = 0
# Preper Submission File
#ensemble = stacked_pred *1
submit = pd.DataFrame()
ensemble = 1
submit['id'] = test_ID
submit['SalePrice'] = ensemble
print(submit)
# ----------------------------- Create File to Submit --------------------------------

submit.to_csv('INT638_Group#13_submission.csv', index = False)
submit.head()