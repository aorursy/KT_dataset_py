import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data
train_house_id = train_data['Id']
test_house_id = test_data['Id']
train_data = train_data.drop(columns=['Id'])
test_data = test_data.drop(columns=['Id'])
non_numerical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                          'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                          'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                          'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                          'BsmtExposure', 'BsmtFinType1', 'Heating', 'BsmtFinType2', 'HeatingQC',
                          'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                          'PoolQC', 'Fence', 'SaleType', 'SaleCondition', 'MiscFeature']

continuous_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
                       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                       'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
sns.distplot(train_data['SalePrice'])
correlation = train_data.corr()
sns.heatmap(correlation)
sns.barplot(x=correlation['SalePrice'], y=correlation['SalePrice'].keys(), orient='h')
correlation['SalePrice']
index = 0
for feature in non_numerical_features:
    for key in train_data[feature].value_counts().keys():
        train_data[feature] = train_data[feature].replace(key, index)
        index += 1
    index = 0

for feature in non_numerical_features:
    for key in test_data[feature].value_counts().keys():
        test_data[feature] = test_data[feature].replace(key, index)
        index += 1
    index = 0
train_data.info()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
train_data
train_data.loc[:, 'MSSubClass':'SaleCondition'] = IterativeImputer().fit_transform(train_data.loc[:, 'MSSubClass':'SaleCondition'])
test_data.loc[:, 'MSSubClass':'SaleCondition'] = IterativeImputer().fit_transform(test_data.loc[:, 'MSSubClass':'SaleCondition'])
train_data
from sklearn.preprocessing import MinMaxScaler
normalizer = MinMaxScaler(feature_range=(0, 1))
train_data[continuous_features] = normalizer.fit_transform(train_data[continuous_features])
train_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
X = train_data.loc[:, 'MSSubClass':'SaleCondition']
y = train_data.loc[:, 'SalePrice']
linear = LinearRegression()
linear_predicted = cross_val_score(linear, X, y, cv=10, scoring='r2')
print(f'LinearRegression: {linear_predicted.mean()}')
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=500, alpha=0.9)
gbr_predicted = cross_val_score(gbr, X, y, cv=10, scoring='r2')
gbr_predicted.mean()
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=7)
dtr_predicted = cross_val_score(dtr, X, y, cv=10, scoring='r2')
dtr_predicted.mean()
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=700)
rfr_predicted = cross_val_score(rfr, X, y, cv=10, scoring='r2')
rfr_predicted.mean()
from sklearn.linear_model import Lasso
lr = Lasso(max_iter=10000)
lr_predicted = cross_val_score(lr, X, y, cv=10, scoring='r2')
lr_predicted.mean()
from sklearn.linear_model import Ridge
rr = Ridge(max_iter=10000)
rr_predicted = cross_val_score(rr, X, y, cv=10, scoring='r2')
rr_predicted.mean()
from sklearn.svm import SVR
svr = SVR(kernel='linear', C=500)
svr_predicted = cross_val_score(svr, X, y, cv=10, scoring='r2')
svr_predicted.mean()
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(500, 500), activation='relu', max_iter=3000)
mlp_predicted = cross_val_score(mlp, X, y, cv=10, scoring='r2')
mlp_predicted.mean()
x = ['LR', 'GB', 'DT', 'RF', 'LASSO', 'Ridge', 'SVR', 'NN']
height = [linear_predicted.mean(), gbr_predicted.mean(), dtr_predicted.mean(), rfr_predicted.mean(), lr_predicted.mean(), rr_predicted.mean(), svr_predicted.mean(), mlp_predicted.mean()]
color = ['black', 'red', 'blue', 'purple', 'orange', 'brown', 'green', 'pink']

plt.xlabel('Techniques')
plt.ylabel('Accuracy')
plt.bar(x=x, height=height, color=color)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X = train_data.loc[:, 'MSSubClass':'SaleCondition']
y = train_data.loc[:, 'SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
gbr = GradientBoostingRegressor(n_estimators=500, alpha=0.9)
gbr.fit(X_train, y_train)
predicted = gbr.predict(X_test)
x = range(len(y_test))
plt.scatter(x, y_test, color='red')
plt.ylabel('SalePrice')
plt.plot(x, predicted, 'k--')
plt.legend(['Predicted', 'True'])
plt.show()
print(f'R-Squared: {r2_score(y_test, predicted)}')
X_for_test = train_data.loc[:, 'MSSubClass':'SaleCondition']
y_for_test = train_data.loc[:, 'SalePrice']

gbr = GradientBoostingRegressor(n_estimators=500, alpha=0.9)
gbr.fit(X_for_test, y_for_test)

predicted = gbr.predict(test_data)
plt.title('Unseen_Data')
plt.xlabel('House_ID')
plt.ylabel('SalePrice')
plt.scatter(test_house_id, predicted, color='blue')
plt.show()
sub = pd.DataFrame({
    'Id':test_house_id,
    'SalePrice':predicted
})
sub.to_csv('sample_submission.csv', index=False)
print('File Saved!')