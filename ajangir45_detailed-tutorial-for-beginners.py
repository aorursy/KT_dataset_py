# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing the training and test data sets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#importing other necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#print top 5 values
df_train.head()
#let's summarize the dataset
df_train.describe()
#check the shape of the data set
df_train.shape
df_test.shape
#assigning id column to this variable
test_ID = df_test['Id']
#delete the id column from datasets
del df_train['Id']
del df_test['Id']
#exploring outliers
fig, ax = plt.subplots()
ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#Deleting outliers
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice');
#Deleting outliers
df_train = df_train.drop(df_train[(df_train['TotalBsmtSF']>2800) & (df_train['SalePrice']<600000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(df_train['TotalBsmtSF'], df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()
#scatter plot GarageArea/SalePrice
fig, ax = plt.subplots()
ax.scatter(x = df_train['GarageArea'], y = df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)
plt.show()
#Deleting outliers
df_train = df_train.drop(df_train[(df_train['GarageArea']>1200) & (df_train['SalePrice']<13.0)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(df_train['TotalBsmtSF'], df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#relationship of SalePrice with YearBuilt
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#check the distribution of target variable 
print ("Skew is:", df_train.SalePrice.skew())
plt.hist(df_train.SalePrice, color='blue')
plt.show()
#as the distribution is positively skewed, we will perform log transformation on the target variable to make it normally distributed

df_train['SalePrice'] = np.log(df_train.SalePrice)
print ("Skew is:", df_train['SalePrice'].skew())
plt.hist(df_train['SalePrice'], color='blue')
plt.show()
#Check the distribution 
sns.distplot(df_train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
#concat the train and test dataset into all_data
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
#Correlation map to see how features are correlated with SalePrice
corrmat = all_data.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
# least correlated features
corrmat = all_data.corr()
least_corr_features = corrmat.index[abs(corrmat["SalePrice"])<0.30]
plt.figure(figsize=(10,10))
g = sns.heatmap(all_data[least_corr_features].corr(),annot=True,cmap="RdYlGn")
least_corr_features
#we will remove the least correlated features from the dataset to make faster and more accurate predictions
all_data.drop(columns={'3SsnPorch','BedroomAbvGr','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','EnclosedPorch','KitchenAbvGr','LotArea','LowQualFinSF','MSSubClass','MiscVal','MoSold','OverallCond','PoolArea','ScreenPorch','YrSold'},inplace=True)
# most correlated features
corrmat = all_data.corr()
most_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(all_data[most_corr_features].corr(),annot=True,cmap="RdYlGn")
#check for missing values
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
#it is advisable to remove the columns having high null values (considering the correlation of that variable)
all_data.drop(columns={'PoolQC','MiscFeature','Alley','Fence','FireplaceQu'},inplace=True)
#check for the most and least correlated features
corr = all_data.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
#delete the 'SalePrice' column
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
#there may be no garage, so filling NaN with 0
all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(0)
#no masonry veneer for some houses
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)
#NaN means there is no basement, so filling the null values with 'None'
for col in('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    all_data[col]=all_data.fillna('None')
#there is only one missing value, so filling it with mode
all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
#no basement, so filling it with zero
for col in ('BsmtFinSF1','TotalBsmtSF'):
    all_data[col] = all_data[col].fillna(0)
#replacing null values with 'None' as there may be no garage in the house
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
#filling with the most occurring value
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
#mostly the dataset consists of same value in 'Utilities', so droping this feature
all_data = all_data.drop(['Utilities'], axis=1)
#NA for 'Functional' variable means typical (as per data description)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
#filling it with most frequent value
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
#as there is only one missing value, we will impute it with the most occurring value
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
#imputing with mode
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#there is no garage, so fill it with 0
for col in ('GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#checking if still the data set has null values or not 
null_columns=all_data.columns[all_data.isnull().any()]
all_data[null_columns].isnull().sum()
#performing label encoding for the categorical variables
from sklearn.preprocessing import LabelEncoder
cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#let's check for the skewed features

from scipy import stats
from scipy.stats import norm, skew 
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
#performing BoxCox transformation for the highly skewed features

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
#dummy categorical features
all_data = pd.get_dummies(all_data)
print(all_data.shape)
#get the new train and test dataset
train = all_data[:ntrain]
test = all_data[ntrain:]
train.head()
test.head()
#importing necessary libraries for the modelling part
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#splitting the data for train and test 
y = df_train['SalePrice']
y_train = y
X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(
                                     train, y_train,
                                     test_size=0.25,
                                     random_state=42
                                     )
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
#defining regression function
import time
from sklearn import metrics

def regression(regr,X_test_sparse,y_test_sparse):
    start = time.time()
    regr.fit(X_train_sparse,y_train_sparse)
    end = time.time()
    rf_model_time=(end-start)/60.0
    print("Time taken to model: ", rf_model_time , " minutes" ) 
    
def regressionPlot(regr,X_test_sparse,y_test_sparse,title):
    predictions=regr.predict(X_test_sparse)
    plt.figure(figsize=(10,6))
    plt.scatter(predictions,y_test_sparse,cmap='plasma')
    plt.title(title)
    plt.show()
    
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.log1p(y_test_sparse), np.log1p(predictions))))
#Lasso regression
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

regression(lasso,X_test_sparse,y_test_sparse)
regressionPlot(lasso,X_test_sparse,y_test_sparse,"Lasso Model")
#ElasticNet regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)

regression(ENet,X_test_sparse,y_test_sparse)
regressionPlot(ENet,X_test_sparse,y_test_sparse,"Elastic Net Regression")
#KernelRidge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)

regression(KRR,X_test_sparse,y_test_sparse)
regressionPlot(KRR,X_test_sparse,y_test_sparse,"Gradient Boosting Regression")
#Gradient Boosting Regression
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

score = rmsle_cv(GBoost)

regression(GBoost,X_test_sparse,y_test_sparse)
regressionPlot(GBoost,X_test_sparse,y_test_sparse,"Gradient Boosting Regression")
#XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

score = rmsle_cv(model_xgb)

regression(model_xgb,X_test_sparse,y_test_sparse)
regressionPlot(model_xgb,X_test_sparse,y_test_sparse,"XGBoost")
#Light GBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(model_lgb)

regression(model_lgb,X_test_sparse,y_test_sparse)
regressionPlot(model_lgb,X_test_sparse,y_test_sparse,"LightGBM")
#averaging base models
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
#calculating the average score of all the models
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#we will define a function to evalute the rmsle value
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
#final training and prediction
#stacked regressor
averaged_models.fit(train.values, y_train)
stacked_train_pred = averaged_models.predict(train.values)
stacked_pred = np.expm1(averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))
#XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
#LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''

print('Average RMSLE score:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
