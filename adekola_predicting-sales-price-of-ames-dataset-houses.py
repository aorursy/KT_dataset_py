#imports 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Read in data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.describe()
#train_data.shape

test_data.shape
train_data['SalePrice'].describe()
sns.distplot(train_data['SalePrice'])
corrmat = train_data.corr()

best_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

g = sns.heatmap(train_data[best_corr_features].corr(), annot=True, cmap="RdYlGn")
total = train_data.isnull().sum().sort_values(ascending=False)

percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)

train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)

train_data.isnull().sum().max()
saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]

train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)

train_data = train_data.drop(train_data[train_data['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

sns.distplot(train_data['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_data['SalePrice'], plot=plt)
#applying log transformation

train_data['SalePrice'] = np.log(train_data['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(train_data['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_data['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(train_data['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_data['GrLivArea'], plot=plt)
#data transformation

train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(train_data['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_data['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(train_data['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_data['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

train_data['HasBsmt'] = pd.Series(len(train_data['TotalBsmtSF']), index=train_data.index)

train_data['HasBsmt'] = 0 

train_data.loc[train_data['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

train_data.loc[train_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_data['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(train_data['GrLivArea'], train_data['SalePrice']);
#scatter plot

plt.scatter(train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], train_data[train_data['TotalBsmtSF']>0]['SalePrice']);
train_data = pd.get_dummies(train_data)

train_data.shape
train_data.head(10)
#marshall the army!!

import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression,Ridge, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns
y = train_data.SalePrice
train_data = train_data.drop("SalePrice", axis=1)

train_data.shape
#split the data

X_train, X_test, y_train, y_test = train_test_split(train_data, y,test_size=0.3, random_state=0)
n_folds = 10

from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold

scorer = make_scorer(mean_squared_error,greater_is_better = False)

def rmse_CV_train(model):

    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train_data.values)

    rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring =scorer,cv=kf))

    return (rmse)

def rmse_CV_test(model):

    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train_data.values)

    rmse = np.sqrt(-cross_val_score(model,X_test,y_test,scoring =scorer,cv=kf))

    return (rmse)
regressor = LinearRegression()

regressor.fit(X_train,y_train)

test_predictions = regressor.predict(X_test)

train_predictions = regressor.predict(X_train)

print('rmse on training data',rmse_CV_train(regressor).mean())

print('rmse on test data',rmse_CV_test(regressor).mean())
model_ridge = Ridge()



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_CV_train(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]

    

cv_ridge_test = [rmse_CV_test(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge_test = pd.Series(cv_ridge_test, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
print("training error:", cv_ridge.min())

print("test error:",cv_ridge_test.min())
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
train_error = rmse_CV_train(model_lasso).mean()

test_error = rmse_CV_test(model_lasso).mean()

print("lasso training error:",train_error)

print("lasso test error:",test_error)
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#preprocess test data as well

from sklearn.preprocessing import Imputer



#drop the empty table first, then get dummies



test_data_ = test_data.copy()

#pd.get_dummies(test_data).shape

#test_predictions = model_lasso.predict(test_data_imputed)

#test_data_imputed.shape



total = test_data_.isnull().sum().sort_values(ascending=False)

percent = (test_data_.isnull().sum()/test_data_.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



test_data_ = test_data_.drop((missing_data[missing_data['Total'] > 1]).index,1)

test_data_ = test_data_.drop(test_data_.loc[test_data_['Electrical'].isnull()].index)





#test_data_copy.shape

test_data_ = pd.get_dummies(test_data_)

test_data_.isnull().sum().max()