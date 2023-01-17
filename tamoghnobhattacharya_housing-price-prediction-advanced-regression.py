import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
indices = test.index
nominal_cols = ['MSSubClass', 'MSZoning', 'Street','LandContour',
                     'LotConfig', 'Neighborhood', 'Condition1','Condition2', 'BldgType',
                     'HouseStyle', 'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd',
                     'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType',
                     'SaleType', 'SaleCondition']
ordinal_cols = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
                     'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                     'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual',
                     'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond',
                     'PavedDrive']
continuous_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                        'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
discrete_cols = ['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                      'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                      'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']
corr = data.corr()
top_corr = corr.index[abs(corr["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
sns.heatmap(data[top_corr].corr(),annot=True,cmap="coolwarm")
plt.figure(figsize=(5,5))
sns.distplot(data['SalePrice'], bins=30)
plt.axvline(x=data['SalePrice'].mean(), linewidth=2)
plt.title('Histogram of Sale Prices')
plt.figure(figsize=(5,5))
stats.probplot(data['SalePrice'], plot=plt)
plt.title('Probability plot of Sale Prices')
print(data['SalePrice'].skew())

data['SalePrice'] = np.log(data['SalePrice'])

plt.figure(figsize=(5,5))
sns.distplot(data['SalePrice'], bins=30)
plt.axvline(x=data['SalePrice'].mean(), linewidth=2)
plt.title('Histogram of Sale Prices')
plt.figure(figsize=(5,5))
stats.probplot(data['SalePrice'], plot=plt)
plt.title('Probability plot of Sale Prices')
print(data['SalePrice'].skew())
data = data.drop(columns=['Alley','PoolQC','Fence','MiscFeature'])
test = test.drop(columns=['Alley','PoolQC','Fence','MiscFeature'])
categorical_cols = nominal_cols + ordinal_cols
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])
numerical_cols = continuous_cols + discrete_cols
for col in numerical_cols:
    data[col] = data[col].fillna(data[col].mean())
    test[col] = test[col].fillna(test[col].mean())
final = pd.concat([data,test])
skew = final[numerical_cols].skew().sort_values(ascending=False)
skew = skew.to_frame(name='Skew')
skew_high = skew[(skew['Skew']>0.5) | (skew['Skew']<-0.5)]
skew_cols = skew_high.index
l = 0.15
for col in skew_cols:
    final[col] = boxcox1p(final[col], l)
le = LabelEncoder()
for col in ordinal_cols:
    final[col] = le.fit_transform(final[col])
i=0
for col in nominal_cols:
    df1 = pd.get_dummies(final[col], drop_first=True)
    final = final.drop(columns=[col])
    if i==0:
        df = df1.copy()
    else:
        df = pd.concat([df,df1], axis=1)
    i+=1
final = pd.concat([final,df], axis=1)
final = final.loc[:,~final.columns.duplicated()]
data = final.iloc[:1460,:]
test = final.iloc[1460:,:]
test = test.drop(columns=['SalePrice'])
X = data.drop(columns=['SalePrice'])
Y = data['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = XGBRegressor()
model.fit(X_train, Y_train)
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
print(mean_absolute_error(Y_train, Y_train_pred))
print(mean_absolute_error(Y_test, Y_test_pred))
preds = model.predict(test).ravel()
preds = np.expm1(preds)
output = pd.DataFrame({"Id": indices,"SalePrice": preds})
output.to_csv('submissions.csv', index=False)