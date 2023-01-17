import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import sklearn.preprocessing as pp

import sklearn.model_selection as ms

import sklearn.metrics as metrics

import json
df = pd.read_csv('../input/train.csv')

df = df.set_index('Id')



sdf = pd.read_csv('../input/test.csv')

sdf = sdf.set_index('Id')

df.head()
price = df.SalePrice

print("Average sale price: " + "${:,.0f}".format(price.mean()))
df = df.drop('SalePrice', axis=1)

all_df = df.append(sdf)

all_df.shape
all_features = 'MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition'.split(',')

numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','TotalBsmtSF','Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

categorical_features = [f for f in all_features if not(f in numeric_features)]



(len(all_features), len(categorical_features), len(numeric_features))
numeric_df = all_df[numeric_features]

numeric_df.shape
X = numeric_df.as_matrix()



# Impute missing



imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

imp = imp.fit(X)

X = imp.transform(X)

X.shape
scaler = pp.StandardScaler()

scaler.fit(X)

X = scaler.transform(X)

X[0, :]
def process_categorical(ndf, df, categorical_features):

    for f in categorical_features:

        new_cols = pd.DataFrame(pd.get_dummies(df[f]))

        new_cols.index = df.index

        ndf = pd.merge(ndf, new_cols, how = 'inner', left_index=True, right_index=True)

    return ndf



numeric_df = pd.DataFrame(X)

numeric_df.index = all_df.index

combined_df = process_categorical(numeric_df, all_df, categorical_features)

combined_df.head()
X = combined_df.as_matrix()

X.shape
#PCA

from sklearn.decomposition import PCA



test_n = df.shape[0]



pca = PCA()

pca.fit(X[:test_n,:], price)

X = pca.transform(X)

X.shape
X_train = X[:test_n,:]

X_train, X_val, y_train, y_val = ms.train_test_split(X_train, price, test_size=0.3, random_state=0)

X_test = X[test_n:,:]



(X_train.shape, X_val.shape, X_test.shape)
from sklearn import linear_model



lr = linear_model.LinearRegression()

lr.fit(X_train, y_train)
def print_score(alg, score, params):

    print('%s score is %f with params %s' % (alg, score, json.dumps(params)))
for a in np.arange(151., 152., 0.1):

    lasso = linear_model.Lasso(alpha=a, max_iter=2000)

    lasso.fit(X_train, y_train)

    print_score('Lasso', lasso.score(X_val, y_val), {'alpha': a})
lasso = linear_model.Lasso(alpha=151.7, max_iter=2000)

lasso.fit(X_train, y_train)
for a in np.arange(25., 26., 0.05):

    ridge = linear_model.Ridge(alpha=a, max_iter=2000)

    ridge.fit(X_train, y_train)

    print_score('Lasso', ridge.score(X_val, y_val), {'alpha': a})
ridge = linear_model.Ridge(alpha=25.5, max_iter=2000)

ridge.fit(X_train, y_train)
import xgboost as xgb



params = {'eval_metric':'rmse'}

xm = xgb.DMatrix(X_train, label=y_train)

xmodel = xgb.train(params, xm)

xg_y_pred = xmodel.predict(xgb.DMatrix(X_val))
print('XGBoost score is %f' % metrics.r2_score(y_val, xg_y_pred))

print('Ridge score is %f' % ridge.score(X_val, y_val))

print('Lasso score is %f' % lasso.score(X_val, y_val))

print('Linear Regression score is %f' % lr.score(X_val, y_val))



best = ridge
from sklearn.metrics import mean_squared_error



y_val_pred = best.predict(X_val)

mse = mean_squared_error(y_val, y_val_pred)

print('ridge mean squared error is %s' % \

      '{:,.2f}'.format(mse))



bmse = mean_squared_error(y_val, xg_y_pred)

print('xgboost mean squared error is %s' % \

      '{:,.2f}'.format(bmse))
def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))



print("Ridge RMSLE is %f" % rmsle(y_val_pred, y_val))

print("XGBoost RMSLE is %f" % rmsle(xg_y_pred, y_val))
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



fig, ax = plt.subplots()



ax.plot(y_val, y_val_pred, 'b.')

ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

ax.set_title('Ridge')

plt.show()
fig, ax = plt.subplots()



ax.plot(y_val, xg_y_pred, 'b.')

ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

ax.set_title('XGBoost')

plt.show()
best.fit(X[:test_n, :], price)

y_submit = best.predict(X_test)

y_submit[y_submit < 0] = 1.

sdf['SalePrice'] = y_submit

sdf.to_csv('submission.csv', columns = ['SalePrice'])



xmodel = xgb.train(params, xgb.DMatrix(X[:test_n, :], label=price))

y_submit = xmodel.predict(xgb.DMatrix(X_test))

y_submit[y_submit < 0] = 1.

sdf['SalePrice'] = y_submit

sdf.to_csv('xg_submission.csv', columns = ['SalePrice'])