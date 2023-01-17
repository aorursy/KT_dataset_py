#performing imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy import stats
import warnings
import missingno as msno
from scipy.stats import boxcox,skew,norm
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
%matplotlib inline
#reading the trainset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#Taking a look at the columns and shape of the trainset
print("Columns of trainset: ",train.columns)
print("Shape of the trainset: ",train.shape)
#printing the head of the trainset
train.head()
#listing the numeric features
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns
#listing the categorical features
categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns
#the ID column is only for indexing and won't help much in training the ML model.So it's safe to delete it

train_ID = train['Id']
test_ID = test['Id']

train = train.drop("Id",axis=1,inplace=False)
test = test.drop("Id",axis=1,inplace=False)

#printing the shapes of train set and test set just to verify that the ID column has been dropped
print("Trainset shape ",train.shape)
print("Testset shape ",test.shape)
train['SalePrice'].describe()
#visualising a histogram of the target variable to see it's distribution
sns.distplot(train['SalePrice'])
#calculating skewness and kutrosis for SalePrice
print("Skewness: ",train['SalePrice'].skew())
print("Kurtosis: ",train['SalePrice'].kurt())
#creating a correlation matrix
corr_mat = train.corr()
print(corr_mat['SalePrice'].sort_values(ascending=False),'\n')
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corr_mat,square=True,vmax=0.8)
k= 11
cols = corr_mat.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#scatter plot garagecars/saleprice. this is not a numerical variable but a dummy variable
var = 'GarageCars'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
#bos plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
f, ax = plt.subplots(figsize=(12,9))
fig = sns.boxplot(x=var, y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
#box plot yearbuilt/saleprice
var = 'YearBuilt'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
f,ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=800000)
plt.xticks(rotation=90)
#scatterplot
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(train[cols],size=3)
plt.show()
fig, ax = plt.subplots(figsize=(16,9))
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()
#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index) #creating a virtual box to corner these
                                                                                        #outliers

#Check the graphic again
fig, ax = plt.subplots(figsize=(16,9))
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
fig, ax = plt.subplots(figsize=(16,9))
ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()
#histogram and normal probability plot
sns.distplot(train['SalePrice'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['SalePrice'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('SalePrice')

fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()
#applying log transformation
train['SalePrice'] = np.log(train['SalePrice']+1)
#transformed histogram and normal probability plot
sns.distplot(train['SalePrice'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['SalePrice'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('SalePrice')

fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()
#histogram and normal probability plot
sns.distplot(train['GrLivArea'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['GrLivArea'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('GrLivArea')

fig = plt.figure()
res = stats.probplot(train['GrLivArea'],plot=plt)
plt.show()
#applying log transformation
train['GrLivArea'] = np.log(train['GrLivArea']+1)
#transformed histogram and normal probability plot
sns.distplot(train['GrLivArea'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['GrLivArea'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('GrLivArea')

fig = plt.figure()
res = stats.probplot(train['GrLivArea'],plot=plt)
plt.show()
#histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['TotalBsmtSF'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('TotalBsmtSF')

fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'],plot=plt)
plt.show()
#transformed histogram and normal probability plot
sns.distplot(np.log(train['TotalBsmtSF']+1),fit=norm)
fig = plt.figure()
res = stats.probplot(np.log(train['TotalBsmtSF']+1),plot=plt)

#Note:This doesn't seem well
train['TotalBsmtSF'],maxlog = boxcox(train['TotalBsmtSF']+1)

#histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'],fit=norm)

#getting fitted parameters used by the function
(mu,sigma) = norm.fit(train['TotalBsmtSF'])
print('\n mu = {0:.2f} and sigma = {1:.2f}'.format(mu,sigma))

#plotting the distribution
plt.legend(['Normal dist. ($\mu=$ {0:.2f} and $\sigma = ${1:.2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.xlabel('TotalBsmtSF')

fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'],plot=plt)
plt.show()
#working with the missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending=False)
missing_vals = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_vals.head(30)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
data_na = missing_vals[:20]
sns.barplot(x=data_na.index, y=data_na.Percent)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
train['PoolQC'] = train['PoolQC'].fillna('None')
test['PoolQC'] = test['PoolQC'].fillna('None')
train['MiscFeature'] = train['MiscFeature'].fillna('None')
test['MiscFeature'] = test['MiscFeature'].fillna('None')
train['Alley'] = train['Alley'].fillna('None')
test['Alley'] = test['Alley'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
test['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')
#filling the lot frontage with the median value. A good trick is to use the median value of the neighbourhood rather than
#lotfrontage, so that it can be realistic in regards to other parameters as well.
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#A thing to note is why train and test be treated like the same but kept different. This is because the test data doesn't leak
#in the training set and your model doesn't 'know' your test data before testing
for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')
for col in ('GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['MasVnrType'] = train['MasVnrType'].fillna('None')
for col in('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
train.shape
test.shape
#Transforming some numerical variables that are really categorical

#MSSubClass = The building class
train['MSSubClass'] = train['MSSubClass'].apply(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)

#Changing OverallQual into categorical variable
train['OverallQual'] = train['OverallQual'].astype(str)
test['OverallQual'] = test['OverallQual'].astype(str)

#Changing OverallCond into categorical variable
train['OverallCond'] = train['OverallCond'].astype(str)
test['OverallCond'] = test['OverallCond'].astype(str)

#Changing year sold and month sold into categorical features
train['YrSold'] = train['YrSold'].astype(str)
test['YrSold'] = test['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)
test['MoSold'] = test['MoSold'].astype(str)
#Label Encoding some categorical variables so that their information can be used
categorical_features = train.select_dtypes(include=[np.object])
cols = list(categorical_features.columns)

#process all the columns, apply label encoding
for c in cols:
    lbl_train = LabelEncoder()
    lbl_train.fit(list(train[c].values))
    train[c] = lbl_train.transform(list(train[c].values))
    
    lbl_test = LabelEncoder()
    lbl_test.fit(list(test[c].values))
    test[c] = lbl_test.transform(list(test[c].values))    
numeric_feats = train.dtypes[train.dtypes!="object"].index

#check skew of all features
skewed_feats = train[numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
print("Skew Features")
skewness = pd.DataFrame({'Skew' : skewed_feats})
print(skewness)
skewness = skewness[abs(skewness.Skew) > 0.65]
print("Features that can be skewed in the dataset are {0}".format(skewness.shape[0]))

skewed_features = skewness.index
for feat in skewed_features:
    train[feat] = np.log(train[feat]+1)
    #print("Lambda for maxlog for {0} is {1}. ".format(feat,maxlog))
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train,test)).reset_index(drop=True)
all_data.drop(['SalePrice'],axis=1,inplace=True)
print(all_data.shape)
all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]
train.shape
test.shape
train.head()
#Import libraries
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,train.values,y_train,scoring='neg_mean_squared_error',cv=kf))
    return(rmse)
lasso_reg = make_pipeline(RobustScaler(),Lasso(random_state=42))
enet_reg = make_pipeline(RobustScaler(),ElasticNet(random_state=42))
krr_reg = KernelRidge(kernel="polynomial") #using krr instead of SVR because it's much faster on medium sized datasets
gboost_reg = GradientBoostingRegressor(loss='huber',random_state=42)
xgb_reg = xgb.XGBRegressor(random_state=42)
score = rmsle_cv(lasso_reg)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(enet_reg)
print("\nElastic Net score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(xgb_reg)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
xgb_reg.fit(train,y_train)
xgb_train_pred = xgb_reg.predict(train)
xgb_pred = np.expm1(xgb_reg.predict(test))
print(rmsle(y_train,xgb_train_pred))
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = xgb_pred
sub.to_csv('submission.csv',index=False)
xgb_pred.shape
