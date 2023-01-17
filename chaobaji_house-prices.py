import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
house_data = pd.read_csv('../input/train.csv')
#找到需要预测的值SalePrice

house_data.columns
house_data.head()
#house_data.isnull().sum()

house_data.count()
house_data = house_data.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1)

house_data.head()
fix_feature = ['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical',

          'GarageType','GarageFinish','GarageQual','GarageCond']

from sklearn.impute import SimpleImputer

for feature_name in fix_feature:

    imputer = SimpleImputer(strategy='most_frequent')

    house_data[[feature_name]] = imputer.fit_transform(house_data[[feature_name]])
dtype_group = house_data.columns.to_series().groupby(house_data.dtypes).groups

dtype_group
object_feature = house_data.select_dtypes(include=['object']).columns.values

object_feature
var = 'BsmtQual'

data = pd.concat([house_data['SalePrice'], house_data[var]], axis=1)

f, ax = plt.subplots(figsize=(26, 12))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
from sklearn.preprocessing import LabelEncoder

for feature_name in object_feature:

    encoder = LabelEncoder()

    house_data[feature_name] = encoder.fit_transform(house_data[feature_name])

house_data.head()
#取出跟预测值相关的前几个特征

corr = house_data.corr()

k_feature = corr.nlargest(10, 'SalePrice')

feature = k_feature.index[0:].values

feature
#看图，这些特征与预测价格的相关度系数都>0.5

plt.figure(figsize=(14,7))

sns.heatmap(data=house_data[feature].corr(), annot=True)
from sklearn.preprocessing import StandardScaler

key_feature = k_feature.index[1:].values

train_X = house_data[key_feature]

train_y = house_data["SalePrice"]



scaler = StandardScaler()

scaler.fit(train_X)

train_X_fit = scaler.transform(train_X)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import numpy as np



X_train,X_test, y_train, y_test = train_test_split(train_X_fit, train_y, test_size=0.33, random_state=42)



clf = RandomForestRegressor(n_estimators=400)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

lin_mse = mean_squared_error(y_test,y_pred)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
house_data_test = pd.read_csv('../input/test.csv')

test_X = house_data_test[key_feature]

for feature_name in key_feature:

    imputer = SimpleImputer(strategy='most_frequent')

    test_X[[feature_name]] = imputer.fit_transform(test_X[[feature_name]])
test_X_fit = scaler.transform(test_X)

test_y = clf.predict(test_X_fit)

test_y
prediction = pd.DataFrame(test_y, columns=['SalePrice'])

result = pd.concat([ house_data_test['Id'], prediction], axis=1)

result.columns
result.to_csv('../predictions.csv', index=False)
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly",degree=3, C=1000, epsilon=0.1)

svm_poly_reg.fit(X_train, y_train)



y_pred_svm = svm_poly_reg.predict(X_test)

lin_mse_svm = mean_squared_error(y_test,y_pred_svm)

lin_rmse_svm = np.sqrt(lin_mse_svm)

lin_rmse_svm
from sklearn.ensemble import BaggingRegressor

bag_reg = BaggingRegressor(n_estimators = 70, random_state=42)

bag_reg.fit(X_train,y_train)

y_pred_bag = bag_reg.predict(X_test)

lin_mse_bag = mean_squared_error(y_test,y_pred_bag)

lin_rmse_bag = np.sqrt(lin_mse_bag)

lin_rmse_bag
test_y_bag = bag_reg.predict(test_X_fit)

test_y_bag
prediction_bag = pd.DataFrame(test_y_bag, columns=['SalePrice'])

result_bag = pd.concat([ house_data_test['Id'], prediction_bag], axis=1)

result_bag.columns
result_bag.to_csv('../predictions_bag.csv', index=False)