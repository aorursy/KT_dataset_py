import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# importing some libraries for visulizations
import matplotlib.pyplot as plt
import seaborn as sns

# importing sklearn to select the model that will be fitting out data into
# we will train_test_split to divide the data
# we will use cross_val_score to determine best accuracy 
from sklearn.model_selection import train_test_split, cross_val_score

# import the data into dataframes using pandas library
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
train.shape, test.shape
cols = {}
uniqueCols =[]
for col in test.columns:
    if col not in cols:
        cols[col]=1
    else:
        cols+=1
for col in train.columns:
    if col not in cols:
        uniqueCols.append(col)

print( uniqueCols)
    
#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
sns.distplot(train['SalePrice'], bins=20, rug=True)

print("Skewness: %0.2f" %train['SalePrice'].skew())
print("Kurtosis: %0.2f" %train['SalePrice'].kurt())
corrmat = train.corr()
plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);

corrmat = train.corr()
# extracting the relevant features
filteredCorrMat_features = corrmat.index[abs(corrmat['SalePrice'])>=0.5]
plt.figure(figsize=(12,12))
# performing corr on the chosen features and presenting it on the heatmap
sns.heatmap(train[filteredCorrMat_features].corr(),annot=True,cmap='winter')
sns.barplot(train.OverallQual,train.SalePrice)

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
totalMissing = all_data.isnull().sum().sort_values(ascending=False)
percentage = ((all_data.isnull().sum()/all_data.isnull().count())*100).sort_values(ascending=False)

missingData = pd.concat([totalMissing,percentage],axis=1,keys=['Total','Percentage'])
missingData.head(20)
plt.subplots(figsize=(15,20))
plt.xticks(rotation='90')
sns.barplot(x=totalMissing.index[:24],y=percentage[:24])
plt.xlabel('features')
plt.ylabel('percentage of missing data')
plt.title('percent of missing data by feature')
plt.show()
# columns to be dropped
columnsToDrop = missingData[missingData['Percentage']>50].index

all_data = all_data.drop(columnsToDrop, axis=1)
# test = test.drop(columnsToDrop, axis=1)
print(all_data.shape)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 
            'BsmtFinType1', 'BsmtFinType2','BsmtFullBath', 'BsmtHalfBath',
            'GarageType', 'GarageFinish', 'GarageQual', 'BsmtUnfSF','BsmtFinSF1','BsmtFinSF2',
            'GarageCond', 'FireplaceQu', 'MasVnrType', 'Exterior2nd'):
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')
#GarageYrBlt replacing missing data with 0
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)

# NA most likely means no masonry veneer for these houses. We can fill in 0
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# let's drop YrSold since it's also not correlated with 'SalePrice'
all_data = all_data.drop('YrSold', axis=1)

# Electrical has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# 'RL' is by far the most common value. So we can fill in missing values with 'RL'
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
# Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling.
all_data = all_data.drop(['Utilities'], axis=1)

# data description says NA means typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#  Replacing missing data with 0 (Since missing in this case would imply 0.)
for col in ('TotalBsmtSF', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
#  Replacing missing data with the most common
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data.isnull().sum().sort_values(ascending=False) #check

from pandas.api.types import is_numeric_dtype
def remove_outliers(df):
    low = .05
    high = .9
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df

remove_outliers(all_data).head()
plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


#Deleting outliers
tempTrain = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

plt.scatter(x = tempTrain['GrLivArea'], y = tempTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
from scipy import stats
from scipy.stats import norm

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function 
# mu is the mean across the population (more accurately, given data)
# and sigma is the standard deviation across the population
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot (probability plot)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


#Appling log(1+x) to all elements of 'SalePrice'
y_train = train["SalePrice"] = np.log1p(train["SalePrice"])

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
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

from scipy.stats import skew 

# extracting numerical features
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
numeric_features = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :numeric_features})
skewness.head(10)

print(all_data.shape)
highly_skewed = ['PoolArea','LotArea','KitchenAbvGr','ScreenPorch']

from scipy.special import boxcox1p
lam = 0.15
for feat in highly_skewed:
    all_data[feat] = boxcox1p(all_data[feat], lam)
all_data = pd.get_dummies(all_data)

train = all_data[:ntrain]
test = all_data[ntrain:]

# train.drop('SalePrice',axis=1,inplace=True)
# train['SalePrice']
print(train.shape, test.shape)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from xgboost import XGBRegressor


# train= train.drop(train.index[[0,1]],axis=0)
print(y_train.shape, train.shape)

X_train,X_test,y_train2,y_test = train_test_split(train.values,y_train,test_size = 0.3,random_state= 0)
X_train.shape,X_test.shape,y_train2.shape,y_test.shape


# Scoring - Root Mean Squared Error
def rmse_CVscore(model,X,y):
    return np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.05, random_state=1))

score = rmse_CVscore(lasso,X_train,y_train2)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.05, l1_ratio=.9, random_state=3))

score = rmse_CVscore(ENet,X_train,y_train2)
print("\nElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


GBoost = GradientBoostingRegressor(n_estimators=1000,max_depth=4,
                                   learning_rate=0.05,
                                   max_features='sqrt',
                                   loss="huber",random_state =5)
score = rmse_CVscore(GBoost,X_train,y_train2)
print("\nGradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# create pipeline
# my_pipeline = make_pipeline(
#     SimpleImputer(),
# XGBR = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state =7,max_depth=3)
# )

# score = rmse_CVscore(XGBR,X_train,y_train2)
# print("\nXGBRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=1000)

score = rmse_CVscore(model_lgb, X_train, y_train2)
print("\nLightGBM Regressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # creating clones of the original models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # fitting our data to the models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #making predictions on our fitted models and averaging them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (ENet,GBoost, lasso, model_lgb))

score = rmse_CVscore(averaged_models,X_train, y_train2)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
print("test shape: {}, train shape: {}".format(test.shape, y_train.shape))

# train=train.drop(train.index[[0,4]],axis=0)

averaged_models.fit(train, y_train)
train_pred = averaged_models.predict(train)
avg_pred = np.expm1(averaged_models.predict(test))

print(rmse(y_train, train_pred))
# test['Id'].shape
# avg_pred.shape
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = avg_pred
sub.to_csv('submission.csv',index=False)
#train.shape
#test.shape