# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
%matplotlib inline
# bring my dataset
df_train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test_ID = test['Id']
df_train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
# Get all Columns name
df_train.columns
df_train.shape
test.shape
# descriptive statistice summary
df_train["SalePrice"].describe()
# Create Histrogram
sns.distplot(df_train["SalePrice"])
# Scatter plot GrLivArea and SalePrice
var = 'GrLivArea'
df_train.plot.scatter(x=var, y="SalePrice")

# Scatter plot GrLivArea and SalePrice
var = 'TotalBsmtSF'
df_train.plot.scatter(x=var, y="SalePrice")

# Box Plot overallqual/saleprice
var = 'OverallQual'
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='SalePrice', data=df_train)
fig.axis(ymin=0, ymax=800000)
df_train.corr()
cormat = df_train.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(cormat, vmax=1, square=True)
# SalePrice correlation matrix
k = 10 # number of variables for heatmap
cols = cormat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size':10},yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show()
fig, ax = plt.subplots()
ax.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
# Deleting Outliers
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

# check the graph again
fig, ax = plt.subplots()
ax.scatter(x=df_train['GrLivArea'], y=df_train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
from scipy import stats
from scipy.stats import norm, skew

sns.distplot(df_train["SalePrice"], fit=norm)

(mu, sigma) = norm.fit(df_train["SalePrice"])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# Get also thr QQ-Plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
# We use numpy function log1p which applies log(1+x) to all elements of the columns
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

#Chek the new distribution
sns.distplot(df_train["SalePrice"], fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_train["SalePrice"])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# Get also thr QQ-Plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
train = df_train
ntrain = train.shape[0]
ntest =  test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("All Data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na =  all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({"Missing Ratio": all_data_na})
missing_data.head(20)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None") 
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None") 
all_data["Alley"] = all_data["Alley"].fillna("None") 
all_data["Fence"] = all_data["Fence"].fillna("None") 
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None") 
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data["MSZoning"].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
# MSSubClass has numeric values but they are categorical data
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


# Changing OverallCond into a catgorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data["TotalBsmtSF"].value_counts()
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical data
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
all_data.shape
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(100)
plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice'])
print(all_data.shape)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to  box cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
all_data.columns
all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test  = all_data[ntrain:]
all_data.shape
2917/2
# import libraries

from sklearn.model_selection import KFold, cross_val_score, train_test_split,StratifiedKFold
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing  import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Validation Function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
score = rmsle_cv(Lasso())
# print("\nLinear Regression score: {:.4f}".format(score))
print("\n Linear Regression Score: Mean = {:.4f} and Std = {:4f}\n".format(score.mean(), score.std()))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet  = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=.9, random_state=3, tol=0.0001))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, x, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(x,y)
        
        return self
    
    def predict(self, x):
        predictions = np.column_stack([
            
            model.predict(x) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
        
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

averaged_models = AveragingModels(models = (ENet, model_xgb, KRR, lasso))

averaged_models.fit(train.values, y_train)
averaged_train_pred = averaged_models.predict(train.values)
averaged_test_pred = np.expm1(averaged_models.predict(test.values))
score = rmse(y_train, averaged_train_pred)
print(" Averaged base models score:{} \n".format(score))
sub = pd.DataFrame()
sub["Id"] = test_ID
sub["SalePrice"] = averaged_test_pred
sub.to_csv("submission.csv", index=False)