import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
d_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

d_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print('*'*100,'\n')

print("About the Dataset".center(50,'*'),'\n')

print('''The Dataset is split into test and train datasets''')

print('\n\n')

print("The Training Dataset has %d rows and %d columns"%(d_train.shape[0],d_train.shape[1]))



print("The Test Dataset has %d rows and %d columns"%(d_test.shape[0],d_test.shape[1]))

print('\n','*'*100)
d_train = d_train.drop('Id',axis=1)

d_test = d_test.drop('Id',axis =1)
nans = pd.concat([d_train.isnull().sum(), (d_train.isnull().sum() / d_train.shape[0])*100 , d_test.isnull().sum(), (d_test.isnull().sum() / d_test.shape[0])*100], axis=1, keys=['Train', 'Percentage', 'Test', 'Percentage'],sort = False)

nans

d_test = d_test.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
d_train = d_train.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
corr = d_train.corr().abs()

f,ax = plt.subplots(figsize=(20,18))

sns.heatmap(corr,square =True,ax =ax,annot=True)

plt.show()
upper = corr.where(np.tril(np.ones(corr.shape), k=0).astype(np.bool))

high_cor = upper[(upper>0.70)&(upper<1)]

f,ax = plt.subplots(figsize=(20,18))

sns.heatmap(high_cor,square=True,annot=True,ax=ax)

plt.show()
Pred_corr = corr.nlargest(10,'SalePrice')

Pred_corr['SalePrice']
d_train.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1, inplace=True)

d_train['TotalBsmtSF'] = d_train['TotalBsmtSF'].fillna(0)

d_train['1stFlrSF'] = d_train['1stFlrSF'].fillna(0)

d_train['2ndFlrSF'] = d_train['2ndFlrSF'].fillna(0)

d_train['TotalSF'] = d_train['TotalBsmtSF'] + d_train['1stFlrSF'] + d_train['2ndFlrSF']

d_train.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

d_train.drop(['GarageCars'], axis=1, inplace=True) 
d_test.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis=1, inplace=True)

d_test['TotalBsmtSF'] = d_test['TotalBsmtSF'].fillna(0)

d_test['1stFlrSF'] = d_test['1stFlrSF'].fillna(0)

d_test['2ndFlrSF'] = d_test['2ndFlrSF'].fillna(0)

d_test['TotalSF'] = d_test['TotalBsmtSF'] + d_test['1stFlrSF'] + d_test['2ndFlrSF']

d_test.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

d_test.drop(['GarageCars'], axis=1, inplace=True)
d_train.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'MasVnrType', 'Heating', 'LowQualFinSF',

            'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

            'Functional', 'GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'WoodDeckSF',

            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

            'MiscVal'], axis=1, inplace=True)
d_test.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'MasVnrType', 'Heating', 'LowQualFinSF',

            'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

            'Functional', 'GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'WoodDeckSF',

            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

            'MiscVal'], axis=1, inplace=True)
d_train = d_train.drop(['MoSold'],axis=1)
d_test = d_test.drop(['MoSold'],axis=1)
d_train.LotFrontage=d_train.LotFrontage.fillna(np.random.randint(68,80))
d_test.LotFrontage=d_test.LotFrontage.fillna(np.random.randint(68,80))
d_train.Electrical=d_train.Electrical.fillna('Mix')
d_train.FireplaceQu=d_train.FireplaceQu.fillna('NA')
d_test.FireplaceQu=d_test.FireplaceQu.fillna('NA')
d_test.MSZoning = d_test.MSZoning.fillna('RM')
d_test.KitchenQual=d_test.KitchenQual.fillna('TA')
d_test.Exterior1st = d_test.Exterior1st.fillna('Other')

d_test.Exterior2nd = d_test.Exterior2nd.fillna('Other')
d_test.SaleType=d_test.SaleType.fillna('Other')
d_test.GarageArea=d_test.GarageArea.fillna(0)
d_train.info()
sns.scatterplot(x='LotFrontage',y='SalePrice',data = d_train)

plt.show()
sns.scatterplot(x='TotalSF',y='SalePrice',data = d_train)

plt.show()
d_train = d_train[(d_train.TotalSF<7500)]
d_train.info()
d_train.OverallQual = d_train.OverallQual.astype('category')
d_train.OverallCond = d_train.OverallCond.astype('category')
d_test.OverallQual = d_test.OverallQual.astype('category')
d_test.OverallCond = d_test.OverallCond.astype('category')
for i in d_train.columns:

        if(d_train[i].dtype=='object'):

            d_train[i]=d_train[i].astype('category')

        else:

            pass
for i in d_test.columns:

        if(d_test[i].dtype=='object'):

            d_test[i]=d_test[i].astype('category')

        else:

            pass
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
X = d_train.drop('SalePrice',axis =1)

y = d_train.SalePrice
X =pd.get_dummies(X,drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
scaler = preprocessing.StandardScaler()

scaler = scaler.fit(X_train)



X_train_transformed = scaler.transform(X_train)





X_test_transformed = scaler.transform(X_test)
X_data = pd.get_dummies(X,drop_first=True)
import statsmodels.api as sm
X_Stats = sm.add_constant(X_train)

X_Stats_Test = sm.add_constant(X_test)
stat_mod =sm.OLS(y_train,X_Stats).fit()
stat_mod.summary()
y_stat_pred = stat_mod.predict(X_Stats)

residuals =stat_mod.resid

print(stat_mod.rsquared)
ax = sns.residplot(y_stat_pred, residuals, lowess = True, color = "b")

ax.set(xlabel='Fitted Value', ylabel='Residuals', title = 'Residual Vs Fitted values PLOT \n')

plt.show()
from sklearn import linear_model

from sklearn.metrics import r2_score,mean_squared_error
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("R2 SCORE OF TRAIN DATA:",round(r2_score(y_pred,y_test)*100,2),'%')

print("RMSE OF TRAIN DATA:",np.sqrt(mean_squared_error(y_test,y_pred)))





from sklearn.linear_model import Ridge

rd=Ridge(alpha=0.3, normalize=True)

rd.fit(X_test,y_test)

print("Ridge Regression Score:",rd.score(X_test,y_test))

rmse = np.sqrt(mean_squared_error(rd.predict(X_train), y_train))

print("RMSE:",rmse)

print("\n********************************\n")

from sklearn.linear_model import Lasso

la=Lasso(alpha=5, normalize=True)

la.fit(X_test,y_test)

print("Lasso Regression score: ",la.score(X_train,y_train))

rmse = np.sqrt(mean_squared_error(la.predict(X_train), y_train))

print("RMSE:",rmse)

print("\n********************************\n")

from sklearn.linear_model import ElasticNet

regr = ElasticNet(alpha = 0.1,random_state=1,max_iter=40000)

regr.fit(X_test, y_test) 

print("Elastic net score: ",regr.score(X_train,y_train))

rmse = np.sqrt(mean_squared_error(regr.predict(X_train), y_train))

print("RMSE:",rmse)

print("\n********************************\n")
final_pred = regr.predict(X_data)
d_t = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
final_file = pd.DataFrame({'Id':d_t.Id[:-1],'SalePrice':final_pred})
final_file.to_csv('Submission.csv',index=False)
#These, are few of the different advanced regression models that are used to predict the house prices.
from sklearn.ensemble import GradientBoostingRegressor
clf =GradientBoostingRegressor(alpha=0.1,n_estimators= 200, max_depth= 4, min_samples_split=2,learning_rate = 0.01, loss= 'ls')



clf.fit(X_train, y_train)

print("Gradient Boosted Score: ",clf.score(X_train,y_train))
rmse = np.sqrt(mean_squared_error(y_test, clf.predict(X_test)))

print("RMSE: %.4f" % rmse)