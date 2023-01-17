#https://anaconda.org/conda-forge/mlxtend
#conda install -c conda-forge mlxtend
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, LassoCV, RidgeCV ,ElasticNetCV, PassiveAggressiveRegressor, HuberRegressor, TheilSenRegressor

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

from mlxtend.regressor import StackingCVRegressor

import missingno as msno
#Let's import our data

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
test.head()
train.select_dtypes(exclude = 'object').columns
train.select_dtypes(include = 'object').columns
num_correlation = train.select_dtypes(exclude='object').corr()
plt.figure(figsize=(20,20))
plt.title('High Correlation')
sns.heatmap(num_correlation > 0.8, annot=True, square=True);
corr = num_correlation.corr()
print(corr['SalePrice'].sort_values(ascending=False))
corr[corr['SalePrice']>0.3].index
train.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True) 
test.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)
# Useless Columns...
train=train.drop(columns=['Street','Utilities']) 
test=test.drop(columns=['Street','Utilities']) 
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', alpha=0.3, ylim=(0,800000));
data = pd.concat([train['SalePrice'], train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', alpha=0.3, ylim=(0,800000));
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', alpha=0.3, ylim=(0,800000));
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(train['SalePrice'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist(bins = 50);
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])
# Remove outliers
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
((all_data.isnull().sum().sort_values(ascending = False)/len(all_data))*100).head(30)
def handle_missing(features):
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    features['GarageCars'] = features['GarageCars'].fillna(0)
    
    # the data description stats that NA refers to "No Pool"
    features["PoolQC"] = features["PoolQC"].fillna("None")

    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
        
    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))    
    return features

all_data = handle_missing(all_data)
all_data.isnull().sum().sort_values(ascending = False).head(5)
#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index


skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
skewed_feats
#skewed_feats = skewed_feats.drop(['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 
 #                                 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'])

for x in skewed_feats:

    print('Skewed features:', x)
    print('\n')
    print('-'*10, '\n')
    sns.distplot(all_data[x], color="b");
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel="SalePrice")
    ax.set(title="SalePrice distribution")
    plt.show();
    print('-'*10, '\n')
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#skewed_feats = skewed_feats.drop(['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 
 #                                 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'])


for x in skewed_feats:

    print('Skewed features:', x)
    print('\n')
    print('-'*10, '\n')
    sns.distplot(all_data[x], color="b");
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel="SalePrice")
    ax.set(title="SalePrice distribution")
    plt.show();
    print('-'*10, '\n')
all_data = all_data.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30',40: 'SubClass_40',
45: 'SubClass_45',50: 'SubClass_50',60: 'SubClass_60',70: 'SubClass_70',
75: 'SubClass_75',80: 'SubClass_80',85: 'SubClass_85',90: 'SubClass_90',
120: 'SubClass_120',150: 'SubClass_150',160: 'SubClass_160',180: 'SubClass_180',
190: 'SubClass_190'}})
all_data = all_data.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30',40: 'SubClass_40',
45: 'SubClass_45',50: 'SubClass_50',60: 'SubClass_60',70: 'SubClass_70',
75: 'SubClass_75',80: 'SubClass_80',85: 'SubClass_85',90: 'SubClass_90',
120: 'SubClass_120',150: 'SubClass_150',160: 'SubClass_160',180: 'SubClass_180',
190: 'SubClass_190'}})
# Some of the non-numeric predictors are stored as numbers; convert them into strings 
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
from pandas.api.types import CategoricalDtype

# Categorical values
# Ordered
all_data["BsmtCond"] = pd.Categorical(all_data["BsmtCond"], categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["BsmtExposure"] = pd.Categorical(all_data["BsmtExposure"], categories=['No','Mn','Av','Gd'],ordered=True)
all_data["BsmtFinType1"] = pd.Categorical(all_data["BsmtFinType1"],categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True)
all_data["BsmtFinType2"] = pd.Categorical(all_data["BsmtFinType2"],categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True)
all_data["BsmtQual"] = pd.Categorical(all_data["BsmtQual"],categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["ExterCond"] = pd.Categorical(all_data["ExterCond"],categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["ExterQual"] = pd.Categorical(all_data["ExterQual"],categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["Fence"] = pd.Categorical(all_data["Fence"],categories=['No','MnWw','GdWo','MnPrv','GdPrv'],ordered=True)
all_data["FireplaceQu"] = pd.Categorical(all_data["FireplaceQu"],categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["Functional"] = pd.Categorical(all_data["Functional"],categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],ordered=True)
all_data["GarageCond"] = pd.Categorical(all_data["GarageCond"],categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["GarageFinish"] = pd.Categorical(all_data["GarageFinish"],categories=['No','Unf','RFn','Fin'],ordered=True)
all_data["GarageQual"] = pd.Categorical(all_data["GarageQual"],categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["HeatingQC"] = pd.Categorical(all_data["HeatingQC"],categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["KitchenQual"] = pd.Categorical(all_data["KitchenQual"],categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
all_data["PavedDrive"] = pd.Categorical(all_data["PavedDrive"],categories=['N','P','Y'],ordered=True)
all_data["PoolQC"] = pd.Categorical(all_data["PoolQC"],categories=['No','Fa','TA','Gd','Ex'],ordered=True)
all_data['TotalSF']=all_data['TotalBsmtSF']  + all_data['2ndFlrSF']
all_data['TotalBath']=all_data['BsmtFullBath'] + all_data['FullBath'] + (0.5*all_data['BsmtHalfBath']) + (0.5*all_data['HalfBath'])
all_data['YrBltAndRemod']=all_data['YearBuilt']+(all_data['YearRemodAdd']/2)
all_data['Porch_SF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])
all_data['Has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFirePlace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFlr']=all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt']=all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['LotArea'] = all_data['LotArea'].astype(np.int64)
all_data['MasVnrArea'] = all_data['MasVnrArea'].astype(np.int64)
all_data = pd.get_dummies(all_data,drop_first=True)

all_data.shape
X = all_data[:train.shape[0]]
y = train.SalePrice
    
Test = all_data[train.shape[0]:]
print(X.shape)
print(Test.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0,test_size=0.25)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
import math
import sklearn.metrics as sklm

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_train)
residuals = y_train.values-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))
p = sns.scatterplot(y_pred,residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-2.5,2.5)
plt.xlim(11,13)
p = sns.lineplot([10,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals, X_train)
lzip(name, test)
p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of error terms/residuals')
plt.figure(figsize=(30,30))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(train.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring = "neg_mean_squared_error" , cv=5))
    return (rmse)
scores = {}

scores['xgboost'] = rmse_cv(XGBRegressor())

scores['LinearRegression'] = rmse_cv(LinearRegression())

scores['SGDRegressor'] = rmse_cv(SGDRegressor())

scores['Lasso'] = rmse_cv(Lasso())

scores['Ridge'] = rmse_cv(Ridge())

scores['PassiveAggressiveRegressor'] = rmse_cv(PassiveAggressiveRegressor())

scores['HuberRegressor'] = rmse_cv(HuberRegressor())

scores['TheilSenRegressor'] = rmse_cv(TheilSenRegressor())

scores['RandomForestRegressor'] = rmse_cv(RandomForestRegressor())

scores['ExtraTreesRegressor'] = rmse_cv(ExtraTreesRegressor())

scores['AdaBoostRegressor'] = rmse_cv(AdaBoostRegressor())

scores['GradientBoostingRegressor'] = rmse_cv(GradientBoostingRegressor())

scores['DecisionTreeRegressor'] = rmse_cv(DecisionTreeRegressor())

scores['ElasticNet'] = rmse_cv(ElasticNet())

scores['SVR'] = rmse_cv(SVR())

model_rmse = pd.DataFrame(scores).mean()
model_rmse = model_rmse.sort_values(ascending=True)
print('Model scores\n{}'.format(model_rmse))
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "rmse mean in fonction of alpha")
plt.xlabel('alpha')
plt.ylabel('rmse (mean)');
cv_ridge.min()
model_ridge = Ridge(alpha = 5).fit(X_train, y_train)
coef = pd.Series(model_ridge.coef_, index = X_train.columns)
print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                    coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model");

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_ridge.predict(X_train), "true":y_train})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter");
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, random_state = 3)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_tr, y_tr, early_stopping_rounds=5, 
             eval_set=[(X_val, y_val)], verbose=False)
my_model.get_params
rmse_cv(XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, importance_type='gain',
       learning_rate=0.05, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=1000, n_jobs=1,
       nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)).mean()
from sklearn.model_selection import GridSearchCV


param_grid={'n_estimators':[100], 
            'learning_rate': [0.1, 0.05, 0.02, 0.01],
            'max_depth':[4,6], 
            'min_samples_leaf':[3,5,9,17],
            'max_features':[1.0,0.3,0.1] }

GB = GradientBoostingRegressor()
clf = GridSearchCV(GB, param_grid, cv=5)
clf.fit(X, y)
print(clf.best_params_)

rmse_cv(GradientBoostingRegressor
        (learning_rate= 0.1, max_depth= 4, max_features= 0.3, min_samples_leaf= 3, n_estimators= 100)).mean()
#GradientBoostingRegressor 

GB = GradientBoostingRegressor(learning_rate= 0.1, max_depth= 4, max_features= 0.3, min_samples_leaf= 3, n_estimators= 100)
GB.fit(X, y)
gb_preds = np.expm1(GB.predict(Test))


# XGBRegressor

model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model_xgb.fit(X_tr, y_tr, early_stopping_rounds=5, 
             eval_set=[(X_val, y_val)], verbose=False)

xgb_preds = np.expm1(model_xgb.predict(Test))

#RIDGE

model_ridge = Ridge(alpha = 10)
model_ridge.fit(X, y)

ridge_preds = np.expm1(model_ridge.predict(Test))
preds = 0.4*gb_preds + 0.3*xgb_preds + 0.3* ridge_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("solution.csv", index = False)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
import math
import sklearn.metrics as sklm

def function(model, model_name):

    train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
    test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
    
    #skew
    train["SalePrice"] = np.log1p(train["SalePrice"])
    
    # Remove outliers
    train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
    train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
    train.reset_index(drop=True, inplace=True)
    
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
    
    #handle missing value
    all_data = handle_missing(all_data)
    
    #log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    
    all_data = all_data.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30',40: 'SubClass_40',
    45: 'SubClass_45',50: 'SubClass_50',60: 'SubClass_60',70: 'SubClass_70',
    75: 'SubClass_75',80: 'SubClass_80',85: 'SubClass_85',90: 'SubClass_90',
    120: 'SubClass_120',150: 'SubClass_150',160: 'SubClass_160',180: 'SubClass_180',
    190: 'SubClass_190'}})
    
    all_data = all_data.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30',40: 'SubClass_40',
    45: 'SubClass_45',50: 'SubClass_50',60: 'SubClass_60',70: 'SubClass_70',
    75: 'SubClass_75',80: 'SubClass_80',85: 'SubClass_85',90: 'SubClass_90',
    120: 'SubClass_120',150: 'SubClass_150',160: 'SubClass_160',180: 'SubClass_180',
    190: 'SubClass_190'}})
    
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
    
    all_data["BsmtCond"] = pd.Categorical(all_data["BsmtCond"], categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["BsmtExposure"] = pd.Categorical(all_data["BsmtExposure"], categories=['No','Mn','Av','Gd'],ordered=True)
    all_data["BsmtFinType1"] = pd.Categorical(all_data["BsmtFinType1"],categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True)
    all_data["BsmtFinType2"] = pd.Categorical(all_data["BsmtFinType2"],categories=['No','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],ordered=True)
    all_data["BsmtQual"] = pd.Categorical(all_data["BsmtQual"],categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["ExterCond"] = pd.Categorical(all_data["ExterCond"],categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["ExterQual"] = pd.Categorical(all_data["ExterQual"],categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["Fence"] = pd.Categorical(all_data["Fence"],categories=['No','MnWw','GdWo','MnPrv','GdPrv'],ordered=True)
    all_data["FireplaceQu"] = pd.Categorical(all_data["FireplaceQu"],categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["Functional"] = pd.Categorical(all_data["Functional"],categories=['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],ordered=True)
    all_data["GarageCond"] = pd.Categorical(all_data["GarageCond"],categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["GarageFinish"] = pd.Categorical(all_data["GarageFinish"],categories=['No','Unf','RFn','Fin'],ordered=True)
    all_data["GarageQual"] = pd.Categorical(all_data["GarageQual"],categories=['No','Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["HeatingQC"] = pd.Categorical(all_data["HeatingQC"],categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["KitchenQual"] = pd.Categorical(all_data["KitchenQual"],categories=['Po','Fa','TA','Gd','Ex'],ordered=True)
    all_data["PavedDrive"] = pd.Categorical(all_data["PavedDrive"],categories=['N','P','Y'],ordered=True)
    all_data["PoolQC"] = pd.Categorical(all_data["PoolQC"],categories=['No','Fa','TA','Gd','Ex'],ordered=True)
    
    all_data['TotalSF']=all_data['TotalBsmtSF']  + all_data['2ndFlrSF']
    
    all_data['TotalBath']=all_data['BsmtFullBath'] + all_data['FullBath'] + (0.5*all_data['BsmtHalfBath']) + (0.5*all_data['HalfBath'])
    
    all_data['YrBltAndRemod']=all_data['YearBuilt']+(all_data['YearRemodAdd']/2)
    
    all_data['Porch_SF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])
    
    all_data['Has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    all_data['HasFirePlace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    all_data['Has2ndFlr']=all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    all_data['HasBsmt']=all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    
    all_data['LotArea'] = all_data['LotArea'].astype(np.int64)
    all_data['MasVnrArea'] = all_data['MasVnrArea'].astype(np.int64)

    all_data = pd.get_dummies(all_data).reset_index(drop=True)
    
    # Remove any duplicated column names
    all_data = all_data.loc[:,~all_data.columns.duplicated()]
    
    X = all_data[:train.shape[0]]
    y = train.SalePrice
    
    Test = all_data[train.shape[0]:]


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .3, random_state=0)
    

    #modeling
    
    if model == Ridge:
   
        model=model(alpha = 10).fit(X_train,y_train)

        y_pred= model.predict(X_test)

    
        print("The  RMSE score achieved with {} is {}".format(model_name, str(math.sqrt(sklm.mean_squared_error(y_test, y_pred)))))
        
        coef = pd.Series(model.coef_, index = X_train.columns)
        
        print()
        print('Our most valuable features with {} are :'.format(model_name))
        print()
        print(coef.sort_values(ascending = False).head(10))
        print()
        print(coef.sort_values(ascending = False).tail(10))
        
        #submission
        model_preds = np.expm1(model.predict(Test))
        
        solution = pd.DataFrame({"id":test.Id, "SalePrice":model_preds})
        solution.to_csv("solution.csv", index = False)
        print()
        print("Submission file created !")

        
    
    elif model == XGBRegressor:

        model = model(n_estimators=1000, learning_rate=0.05).fit(X_train, y_train)
        
        y_pred=model.predict(X_test)
        
        print("The  RMSE score achieved with {} is {}".format(model_name, str(math.sqrt(sklm.mean_squared_error(y_test, y_pred)))))
        
        
        coef = pd.Series(model.feature_importances_, index = X_train.columns)
        
        print()
        print('Our most valuable features with {} are :'.format(model_name))
        print()
        print(coef.sort_values(ascending = False).head(10))
        
        #submission
        model_preds = np.expm1(model.predict(Test))
        
        solution = pd.DataFrame({"id":test.Id, "SalePrice":model_preds})
        solution.to_csv("solution.csv", index = False)
        
        print()
        print("Submission file created !")
        
    elif model == GradientBoostingRegressor:

        model = model(learning_rate= 0.1, max_depth= 4, max_features= 0.3, min_samples_leaf= 3, n_estimators= 100).fit(X_train, y_train)
        
        y_pred=model.predict(X_test)
        
        print("The  RMSE score achieved with {} is {}".format(model_name, str(math.sqrt(sklm.mean_squared_error(y_test, y_pred)))))
        
        
        coef = pd.Series(model.feature_importances_, index = X_train.columns)
        
        print()
        print('Our most valuable features with {} are :'.format(model_name))
        print()
        print(coef.sort_values(ascending = False).head(10))
        
        #submission
        model_preds = np.expm1(model.predict(Test))
        
        solution = pd.DataFrame({"id":test.Id, "SalePrice":model_preds})
        solution.to_csv("solution.csv", index = False)
        
        print()
        print("Submission file created !")
function(Ridge, 'Ridge')
function(XGBRegressor, 'XGBRegressor')
function(GradientBoostingRegressor, 'GradientBoostingRegressor')
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from mlxtend.regressor import StackingCVRegressor

# Setup cross validation folds
kf = KFold(n_splits=12, random_state=42, shuffle=True)

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
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

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
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)
print('xgboost')
xgb_model_full_data = xgboost.fit(X, y)
print('Svr')
svr_model_full_data = svr.fit(X, y)
print('Ridge')
ridge_model_full_data = ridge.fit(X, y)
print('RandomForest')
rf_model_full_data = rf.fit(X, y)
print('GradientBoosting')
gbr_model_full_data = gbr.fit(X, y)
# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.1 * np.expm1(ridge_model_full_data.predict(X))) + \
            (0.2 * np.expm1(svr_model_full_data.predict(X))) + \
            (0.1 * np.expm1(gbr_model_full_data.predict(X))) + \
            (0.1 * np.expm1(xgb_model_full_data.predict(X))) + \
            (0.1 * np.expm1(lgb_model_full_data.predict(X))) + \
            (0.05 * np.expm1(rf_model_full_data.predict(X))) + \
            (0.35 * np.expm1(stack_gen_model.predict(np.array(X)))))

solution2 = pd.DataFrame({"id":test.Id, "SalePrice":blended_predictions(Test)})

solution2.to_csv("solution2.csv", index = False)
import eli5
from IPython.display import display, HTML
function(Ridge, 'Ridge')
def function(train, test):
    
    #skew
    train["SalePrice"] = np.log1p(train["SalePrice"])
    
    # Remove outliers
    train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
    train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
    train.reset_index(drop=True, inplace=True)
    
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
    
    #handle missing value
    all_data = handle_missing(all_data)
    
    #log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    
    all_data = pd.get_dummies(all_data).reset_index(drop=True)
    
    # Remove any duplicated column names
    all_data = all_data.loc[:,~all_data.columns.duplicated()]
    
    X = all_data[:train.shape[0]]
    y = train.SalePrice
    
    Test = all_data[train.shape[0]:]
    
    return X, y, Test

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

X, y, Test = function(train, test)
X = X[['GrLivArea', '1stFlrSF', 'OverallQual', 'Neighborhood_Crawfor', 'LotArea',
       'Functional_Typ', 'Neighborhood_StoneBr', 'Exterior1st_BrkFace', 'MSZoning_FV', 'KitchenQual_Ex',
       'OverallQual', 'KitchenAbvGr', 'Condition1_Artery', 'Street_Grvl', 'SaleType_WD',
       'Neighborhood_NWAmes', 'Neighborhood_IDOTRR', 'SaleCondition_Abnorml', 'Neighborhood_Edwards', 'Functional_Maj2', 'MSZoning_C (all)']]
Test = Test[['GrLivArea', '1stFlrSF', 'OverallQual', 'Neighborhood_Crawfor', 'LotArea',
       'Functional_Typ', 'Neighborhood_StoneBr', 'Exterior1st_BrkFace', 'MSZoning_FV', 'KitchenQual_Ex',
       'OverallQual', 'KitchenAbvGr', 'Condition1_Artery', 'Street_Grvl', 'SaleType_WD',
       'Neighborhood_NWAmes', 'Neighborhood_IDOTRR', 'SaleCondition_Abnorml', 'Neighborhood_Edwards', 'Functional_Maj2', 'MSZoning_C (all)']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .3, random_state=0)
model_ridge = Ridge(alpha = 10).fit(X_train,y_train)
y_pred=model_ridge.predict(X_test)

print("The  RMSE score achieved with is ", str(math.sqrt(sklm.mean_squared_error(y_test, y_pred))))       
model_preds = np.expm1(model_ridge.predict(Test))
        
solution = pd.DataFrame({"id":test.Id, "SalePrice":model_preds})
solution.to_csv("solution.csv", index = False)
