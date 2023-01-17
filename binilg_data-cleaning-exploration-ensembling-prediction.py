import pandas as pd

import numpy as np

pd.options.display.max_rows = 999

pd.options.display.max_columns = 999

import matplotlib.pyplot as plt

import seaborn as sns

#from pylab import rcParams

plt.rcParams['figure.figsize'] = 20, 5

import warnings

warnings.filterwarnings('ignore')

from scipy.special import boxcox1p

from scipy import stats

from scipy.stats import skew

from scipy.stats import boxcox_normmax
train1 = pd.read_csv('../input/train.csv')

test1 = pd.read_csv('../input/test.csv')
train = train1.copy()

test = test1.copy()
test_ID = test["Id"]
train.head()
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
train.isnull().sum() / train.shape[0]
plt.scatter(train[(train["LotArea"]<30000) & (train["LotFrontage"]<200)]["LotArea"],train[(train["LotArea"]<30000) & (train["LotFrontage"]<200)]["LotFrontage"])
train["train"] = 1

test["train"] = 0
full = pd.concat([train,test],axis=0,sort=False)
full.shape
import pylab

# plot the data itself

x = full[(full["LotArea"]<30000) & (full["LotFrontage"]<200)]["LotArea"].values

y = full[(full["LotArea"]<30000) & (full["LotFrontage"]<200)]["LotFrontage"].values

pylab.plot(x,y,'o')



# calc the trendline

z = np.polyfit(x, y, 1)

p = np.poly1d(z)

pylab.plot(x,p(x),"r--")

# the line equation:
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(pd.DataFrame(x),y)
full.loc[full["LotFrontage"].isnull(),"LotFrontage"]= lm.predict(pd.DataFrame(full[full["LotFrontage"].isnull()]["LotArea"]))
full['MSZoning'] = full['MSZoning'].fillna(full['MSZoning'].mode()[0])
ax = sns.countplot(x=full[full["BsmtQual"]=="Ex"]["YearBuilt"])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.axes.get_yaxis().set_visible(False)
plt.figure(figsize=(20,5))

ax = sns.countplot(x=full[full["BsmtQual"]=="Gd"]["YearBuilt"])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.axes.get_yaxis().set_visible(False)
plt.figure(figsize=(20,5))

sns.countplot(x=train[train["BsmtQual"].isnull()]["YearBuilt"])
full['BsmtQual'].fillna(full.groupby('YearBuilt')['BsmtQual'].agg(lambda x:x.value_counts().index[0]))
#dropping columns which has a lot of nulls or whcih doesn't looks to have predictive ability

cols_to_drop = ["Alley","PoolQC","Fence","MiscFeature","Utilities"]
if "Alley" in full.columns:

    full.drop(cols_to_drop,axis=1,inplace=True)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual','GarageCond','MasVnrType','Functional','GarageYrBlt'):

    full[col] = full[col].fillna('None')
full["GarageYrBlt"] = full["GarageYrBlt"].replace("None",-1)
full["GarageYrBlt"] = full["GarageYrBlt"].astype(int)
full['SaleType'] = full['SaleType'].fillna(full['SaleType'].mode()[0])
full['Exterior1st'] = full['Exterior1st'].fillna(full['Exterior1st'].mode()[0])

full['Exterior2nd'] = full['Exterior2nd'].fillna(full['Exterior2nd'].mode()[0])

full['BsmtFinSF1'] = full['BsmtFinSF1'].fillna(0)

full['BsmtFinSF2'] = full['BsmtFinSF2'].fillna(0)

full['TotalBsmtSF'] = full['TotalBsmtSF'].fillna(0)

full['BsmtUnfSF'] = full['BsmtUnfSF'].fillna(0)

full['BsmtFullBath'] = full['BsmtFullBath'].fillna(0)

full['BsmtHalfBath'] = full['BsmtHalfBath'].fillna(0)

full['Electrical'] = full['Electrical'].fillna(full['Electrical'].mode()[0])

full['KitchenQual'] = full['KitchenQual'].fillna(full['KitchenQual'].mode()[0])

full['MasVnrArea'] = full['MasVnrArea'].fillna(0)

full['GarageCars'] = full['GarageCars'].fillna(0)

full['GarageArea'] = full['GarageArea'].fillna(0)
#Feature Engineering

full['TotalSF'] = full['TotalBsmtSF'] + full['1stFlrSF'] + full['2ndFlrSF']

full['Total_Bathrooms'] = (full['FullBath'] + (0.5 * full['HalfBath']) + full['BsmtFullBath'] + (0.5 * full['BsmtHalfBath']))

full['Total_porch_sf'] = (full['OpenPorchSF'] + full['3SsnPorch'] + full['EnclosedPorch'] + full['ScreenPorch'] + full['WoodDeckSF'])

full["Age"] = full["YrSold"] - full["YearBuilt"]

full["Age_Remod"] = full["YrSold"] - full["YearRemodAdd"]

full["Age_Garage"] = full["YrSold"] - full["GarageYrBlt"]
mask = full.Age < 0

full.loc[full.Age<0,"Age"] = 0
full["YrSold"] = full["YrSold"].astype("object")

full["MoSold"] = full["MoSold"].astype("object")

full["MSSubClass"] = full["MSSubClass"].astype("object")
full.loc[full["Age_Garage"]>2000, "Age_Garage"] = -1
full.drop(["YearBuilt","YearRemodAdd","GarageYrBlt"],axis=1,inplace=True)
train = full[full['train']==1]

test = full[full['train']==0]

data_num = train.drop(["Id"],axis=1).select_dtypes(include = ['float64', 'int64'])

numeric_cols = data_num.columns
for i in full.drop(["SalePrice","train"],axis=1).columns:

    if (i in numeric_cols) and (skew(full[i])>0.6) and full[i].value_counts().shape[0]>100:

        print(i,skew(full[i]))

        #print(full[i].value_counts())

        full[i] = np.log1p(full[i])
for i in range(0, len(data_num.columns), 4):

    sns.pairplot(data=data_num,

                x_vars=data_num.columns[i:i+4],

                y_vars=['SalePrice'])
data_obj =  train.drop(["Id"],axis=1).select_dtypes(include = ['object'])
data_obj = data_obj.join(train['SalePrice'])
for i in range(0, len(data_obj.drop('SalePrice',axis=1).columns)):

    plt.figure(i)

    ax = sns.boxplot(x = data_obj.columns[i],y = 'SalePrice', data=data_obj)

    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)

    ax.axes.get_yaxis().set_visible(False)
#one hot encoding

full = pd.get_dummies(full, columns=data_obj.drop("SalePrice",axis=1).columns)
#plotting to see the distribution of visitors 

plt.xlim(40000, 500000)

sns.distplot(train["SalePrice"].values,bins=100)
#plotting to see the distribution of visitors 

plt.xlim(10, 15)

sns.distplot(np.log1p(train["SalePrice"].values),bins=100)
train = full[full['train']==1]

test = full[full['train']==0]
train.drop("train",axis=1,inplace=True)

test.drop("train",axis=1,inplace=True)
#LightGBM

params = {

    'learning_rate':[0.01,0.02],

    'n_estimators':[100,200,300,400,500,550,600,650],

    'num_leaves':[5,6,7,8],

    'boosting_type':['gbdt'],

    'metric':['rmse'],

    'objective':['regression'],

    'max_depth':[5,6,7,8],

    'sub_feature':[0.5,0.6,0.7,0.75,0.8,0.85,0.9],

    'subsample':[0.5,0.6,0.7,0.75,0.8,0.85,0.9],

    'min_child_samples':[6,7,5,10],

    'lambda_l1':[0,1,2,3,4,5,6,7,8,9,10],

    'lambda_l2':[0,1,2,3,4,5,6,7,8,9,10]

}
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV

from sklearn import preprocessing, metrics

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.svm import SVC

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, Ridge, SGDRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR, LinearSVR 
X_train, X_test, y_train, y_test = train_test_split(train.drop(["Id","SalePrice"],axis=1), train["SalePrice"], test_size=0.3, random_state=42)

clf = lgb.LGBMRegressor()
grid = RandomizedSearchCV(clf,params,verbose=1,cv=5,n_jobs=-1,n_iter=100)
#LightGBM

usable_columns = list(set(train.columns) - set(['Id','SalePrice']))

fold_n=5

folds = KFold(n_splits=fold_n, shuffle=True, random_state=2319)

y_pred_lgb = np.zeros(len(test))

y_pred_train =  np.zeros(len(train))

i = 0

rmse = 0

gridParams = {'subsample': 0.8,'sub_feature': 0.75,'objective': 'regression','num_leaves': 8,'n_estimators': 1000,'min_child_samples': 7,'metric': 'rmse','max_depth': 8,'learning_rate': 0.02,'lambda_l2': 5,'lambda_l1': 0,'boosting_type': 'gbdt'}

for fold_, (train_index, valid_index) in enumerate(folds.split(train.drop(["Id","SalePrice"],axis=1),np.log1p(train["SalePrice"]))):

    print("Fold = {}".format(fold_+1))

    train = pd.DataFrame(train)

    y = pd.DataFrame(np.log1p(train["SalePrice"]))

    

    X_t, y_t = pd.DataFrame(train).iloc[train_index][usable_columns],pd.DataFrame(y).iloc[train_index]

    

    trn_data = lgb.Dataset(X_t, label=y_t)

    val_data = lgb.Dataset(train.iloc[valid_index][usable_columns], label=y.iloc[valid_index])

    

    lgb_model = lgb.train(gridParams, trn_data,valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 10)



    y_pred_train[valid_index] = lgb_model.predict(train.iloc[valid_index][usable_columns].values,num_iteration=lgb_model.best_iteration)

    

    y_actuals_lgb = y.iloc[valid_index]

    rmse += np.round(np.sqrt(metrics.mean_squared_error(y_actuals_lgb,y_pred_train[valid_index])),4)

    print("RMSE = ", rmse)

    

    y_pred_lgb += lgb_model.predict(test[usable_columns], num_iteration=lgb_model.best_iteration)/fold_n

print("Mean RMSE = ", rmse/5)

result=pd.DataFrame({'Id':test_ID, 'SalePrice':np.expm1(y_pred_lgb)})

result.to_csv("submission.csv",index=False)
#Linear Regression

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, Ridge, SGDRegressor
#LassoRegression

usable_columns = list(set(train.columns) - set(['Id','SalePrice']))

clf = Lasso(alpha=0.0005)

clf.fit(train[usable_columns],np.log1p(train["SalePrice"]))

y_train_lasso = np.expm1(clf.predict(train[usable_columns]))

y_pred_lasso = np.expm1(clf.predict(test[usable_columns]))

print(np.round(np.sqrt(metrics.mean_squared_error(np.log1p(train["SalePrice"]),np.log1p(y_train_lasso))),4))

result=pd.DataFrame({'Id':test_ID, 'SalePrice':y_pred_lasso})

result.to_csv("submission.csv",index=False)
#RidgeRegression

clf = Ridge(alpha=60)

clf.fit(train[usable_columns],np.log1p(train["SalePrice"]))

y_train_ridge = np.expm1(clf.predict(train[usable_columns]))

y_pred_ridge = np.expm1(clf.predict(test[usable_columns]))

print(np.round(np.sqrt(metrics.mean_squared_error(np.log1p(train["SalePrice"]),np.log1p(y_train_ridge))),4))

result=pd.DataFrame({'Id':test_ID, 'SalePrice':y_pred_ridge})

result.to_csv("submission.csv",index=False)
#Elastic Net

clf = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

clf.fit(train[usable_columns],np.log1p(train["SalePrice"]))

y_train_EN = np.expm1(clf.predict(train[usable_columns]))

y_pred_EN = np.expm1(clf.predict(test[usable_columns]))

print(np.round(np.sqrt(metrics.mean_squared_error(np.log1p(train["SalePrice"]),np.log1p(y_train_EN))),4))

result=pd.DataFrame({'Id':test_ID, 'SalePrice':y_pred_EN})

result.to_csv("submission.csv",index=False)
#Ensembling & Submission

result=pd.DataFrame({'Id':test_ID, 'SalePrice':(0.3*y_pred_lasso+ 0.1*np.expm1(y_pred_lgb) + 0.3*y_pred_EN + 0.3*y_pred_ridge) })

result.to_csv("submission.csv",index=False)