import pandas as pd

import numpy as np

from sklearn.preprocessing import Imputer 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#importing all the liabriaries we will need to boost our model 

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
hp_train = pd.read_csv('../input/traincsv/train.csv')

hp_test = pd.read_csv('../input/testcsv/test.csv')

train_length=1460

saleprice=pd.DataFrame(hp_train.iloc[:,-1])
# #hp_train = pd.read_csv('https://raw.githubusercontent.com/K1TS/mypackage/master/train.csv')

# #hp_test = pd.read_csv('https://raw.githubusercontent.com/K1TS/House-Prices-Advanced-Regression-Techniques/master/test.csv')

#sumple_supmition =pd.read_csv('https://raw.githubusercontent.com/K1TS/House-Prices-Advanced-Regression-Techniques/master/sample_submission.csv')
# We are Concatinating the two datasets so it can be easy to clean data at once for future engineering   

df_all=pd.concat([hp_train.drop(columns=['SalePrice']), hp_test])
# The Shape Of The Two DataSets Combined

df_all.shape
df_all.head()# viewing the dataset to avoid large output space
#We Decided To Cut The Correlation Graph In Half Since The Upper Part Is Really Repetition Of the Lower Triangle  

corr_matrix = hp_train.corr()

mask = np.zeros_like(corr_matrix, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(25, 15))

sns.heatmap(corr_matrix, mask=mask, vmax=0.8, vmin=0.05, annot=True);



#COLUMNS THAT ARE CORRELATED TO THE SALES PRICE AND INVESLY CORRELATED TO SALES PRICE 

df=pd.DataFrame(hp_train.corr()['SalePrice'].sort_values(ascending=False))

df.plot(kind='bar',figsize=(12,5),color='red')

plt.title('Brief Correlation Graph of Features and Sales Price')





fig, axes = plt.subplots(nrows= 3,ncols = 3, figsize=(20,12))



axes[0,0].scatter(hp_train['YearBuilt'], hp_train['SalePrice'], color='orange')

axes[0,1].scatter(hp_train['GarageYrBlt'], hp_train['SalePrice'],  color='green')

axes[0,2].scatter(hp_train['GrLivArea'], hp_train['SalePrice'], color='blue')





axes[1,0].scatter(hp_train['TotRmsAbvGrd'], hp_train['SalePrice'],  color='orange')

axes[1,1].scatter(hp_train['GarageCars'], hp_train['SalePrice'], color='green')

axes[1,2].scatter(hp_train['GarageArea'], hp_train['SalePrice'],  color='blue')



axes[2,0].scatter(hp_train['TotalBsmtSF'], hp_train['SalePrice'],  color='orange')

axes[2,1].scatter(hp_train['1stFlrSF'], hp_train['SalePrice'], color='green')

axes[2,2].scatter(hp_train['OverallQual'], hp_train['SalePrice'],  color='blue')



#Naming Titles Of The Columns  

axes[0,0].set_title('YearBuilt')

axes[0,1].set_title('GarageYrBlt')

axes[0,2].set_title('GrLivArea')



axes[1,0].set_title('TotRmsAbvGrd')

axes[1,1].set_title('GarageCars')

axes[1,2].set_title('GarageArea')





axes[2,0].set_title('TotalBsmtSF')

axes[2,1].set_title('1stFlrSF')

axes[2,2].set_title('OverallQual')
#hp_train.isnull().sum().sort_values(ascending = False).head(20)

# columns that have NaN on the train dataset.

is_null=df_all.isnull().sum().sort_values(ascending=False)

NaN_train=(is_null[is_null>0])

dict(NaN_train)

NaN_train
#THIS ARE THE VISUALS  TO SHOW THE MISSING DATA IN OUR DATASET ..THE COLUMNS WE SORTED IN THE Visualization

plt.figure(figsize=(15, 8))

sns.barplot(NaN_train,NaN_train.index)

plt.title('Missing  Data In The Dataset')



df_all["PoolQC"] = df_all["PoolQC"].fillna("None")



df_all["MiscFeature"] = df_all["MiscFeature"].fillna("None")



df_all["Alley"] = df_all["Alley"].fillna("None")



df_all["Fence"] = df_all["Fence"].fillna("None")



df_all["FireplaceQu"] = df_all["FireplaceQu"].fillna("None")



#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    df_all[col] =df_all[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    df_all[col] = df_all[col].fillna(0)

    

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_all[col] = df_all[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df_all[col] = df_all[col].fillna('None')

    

df_all["MasVnrType"] = df_all["MasVnrType"].fillna("None")

df_all["MasVnrArea"] = df_all["MasVnrArea"].fillna(0)



df_all['MSZoning'] = df_all['MSZoning'].fillna(df_all['MSZoning'].mode()[0])



df_all["Functional"] = df_all["Functional"].fillna("Typ")



df_all['Electrical'] = df_all['Electrical'].fillna(df_all['Electrical'].mode()[0])



df_all['KitchenQual'] = df_all['KitchenQual'].fillna(df_all['KitchenQual'].mode()[0])



df_all['Exterior1st'] = df_all['Exterior1st'].fillna(df_all['Exterior1st'].mode()[0])

df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna(df_all['Exterior2nd'].mode()[0])



df_all['SaleType'] = df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])



df_all['MSSubClass'] = df_all['MSSubClass'].fillna("None")

from scipy import stats

from scipy.stats import norm, skew #for some statistics

(mu, sigma) = norm.fit(saleprice['SalePrice'])
#We Used This Code To Check The skiweness Of The  Salesprice Column 

(mu, sigma) = norm.fit(saleprice['SalePrice'])

sns.distplot(saleprice['SalePrice'],fit=norm)

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
quantile_plot=stats.probplot(saleprice['SalePrice'], plot=plt)
saleprice["SalePrice"] = np.log1p(saleprice["SalePrice"])

y=saleprice

y.head()
(mu, sigma) = norm.fit(saleprice['SalePrice'])

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(saleprice['SalePrice'],fit=norm)

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1, 2, 2)

quantile_plot=stats.probplot(saleprice['SalePrice'], plot=plt)
fat = 'OverallQual'

data = pd.concat([hp_train['SalePrice'], hp_train[fat]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=fat, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('lethabo.png')
plt.figure(figsize=(10,10))

sns.boxenplot(df_all["LotFrontage"],df_all["Neighborhood"])
df_all=df_all.drop(['GarageYrBlt','TotRmsAbvGrd','GarageArea','PoolQC', 'MiscFeature', 'Fence','MiscVal','PoolArea','Utilities'], axis=1)
df_all['TotalBath'] = df_all['FullBath'] + df_all['HalfBath']*0.5 + df_all['BsmtFullBath'] + df_all['BsmtHalfBath']*0.5

df_all['TotalFlrSF'] = df_all['1stFlrSF'] + df_all['2ndFlrSF']

df_all['BsmtFinSF'] = df_all['BsmtFinSF1'] + df_all['BsmtFinSF2']

# Deleting singulars since we have combined above:



singles_to_drop = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2']

for col in singles_to_drop:

    df_all.drop([col], axis =1, inplace = True)


df_all.head()
df_all = pd.get_dummies(df_all)

X_test=df_all.iloc[train_length:,:]

X_train=df_all.iloc[:train_length,:]

X=X_train
df_all.head()
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

import random

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.04, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =random.randint(0,int(2**16)), nthread = -1)

lgb_model = lgb.LGBMRegressor(lcolsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.04, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =random.randint(0,int(2**16)), nthread = -1)
import random

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.04, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =random.randint(0,int(2**16)), nthread = -1)

score = rmsle_cv(lasso)

print("\nLASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(lgb_model)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
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
averaged_models = AveragingModels(models = (GBoost, lasso,KRR,model_xgb))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_models.fit(X,y)
y_average=np.expm1(averaged_models.predict(X_test))
sample2 =pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sales2=pd.DataFrame(y_average,columns=['SalePrice'])

sample2['SalePrice']=sales2['SalePrice']

sample2.head()

sample2.to_csv('avg6.csv',index=False)