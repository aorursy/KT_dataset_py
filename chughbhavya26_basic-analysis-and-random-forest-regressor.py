"""
Importing all the necessary libraries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
"""Getting the dataset"""
df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
df.head()
df.shape
"""
The houses with exterior condition excellent have the highest sale price with brick tile foundation.
We mostly have data of houses which are in typical/avergae condition and the cinder block houses in
poor exterior condition have lowest price.
"""
#plt.figure(figsize=(10,20))
sns.barplot(x="Foundation", y="SalePrice", hue='ExterCond', data=df)

plt.show()
"""FV types of MSZoning have higher sale prices as they begin from 200000 than all other types. Also 
mostly we have data of normal sale condition which have almost all sale prices less than 300000. On the
other hand partial sale condition have higher sale prices which are above 300000 in MSZoning RL type.
"""
sns.swarmplot(x="MSZoning", y="SalePrice", hue='SaleCondition', data=df)
plt.figure(figsize=(10,15))
plt.show()
"""The sale price correlates over a number of features like garage cars and area, overquality etc,
however many features are correlated amongst themselves.
"""
corr = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr)
plt.show()
"""The distribution of Neighbourhood variable shows that we have mostly the data of houses 
in North Ames and then in college creek"""
plt.figure(figsize=(20,10))
sns.countplot(df['Neighborhood'])
plt.show()
"""
Comparing the target variable with some of the area variables. We can see that the variable  
GRLivArea has influences the Sale Price positively while some have not a lot of relation with target variable.
"""
sns.pairplot(data=df,
                  y_vars=['SalePrice'],
                  x_vars=['GarageArea', 'MasVnrArea', 'TotalBsmtSF', 
                          'GrLivArea', 'WoodDeckSF','LotArea'])
plt.tight_layout()
plt.show()
"""Checking to see the distribution of fireplace quality"""

sns.countplot(x= 'FireplaceQu', data=df)
plt.show()
"""
Removing columns with almost all null values like Alley, PoolQC, Fence, MiscFeat
"""
check_na = pd.concat([df.isnull().sum()], axis=1, keys=['Train'])
check_na[check_na.sum(axis=1) > 0]
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
df.drop(['Id', 'GarageYrBlt'], axis=1,inplace=True)
df['FireplaceQu'] = df['FireplaceQu'].fillna(value='No_FP')
df['LotFrontage'] = df['LotFrontage'].fillna(value = df['LotFrontage'].mean())
df['BsmtCond'] =  df['BsmtCond'].fillna(value='No_BS')
df['BsmtQual'] =  df['BsmtQual'].fillna(value='No_BS')
for col in ('BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[col] = df[col].fillna('No_BS')
for col in ('GarageQual', 'GarageCond', 'GarageFinish','GarageType'):
    df[col] = df[col].fillna('No_Garage')
df.dropna(axis=0, inplace=True)
df_all_category = df.select_dtypes(include=['object'])
df = df[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition','SalePrice']]
"""Here I am doing label encoding of only categorical variables."""
le = preprocessing.LabelEncoder()
for col in df.columns.values:
        if df[col].dtypes=='object':
            le.fit(df[col].values)
            df[col]=le.transform(df[col])
X = df.iloc[:,0:74]
Y = df.iloc[:,-1]
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500, random_state=2, oob_score=True)
regressor.fit(X,Y)
print('Out-of-bag score estimate:', (regressor.oob_score_))

"""Discoving the feature importance of various variables and found out that the overall quality
of the hourse if the most important variable.The grpah only contains 25 top variables for predicting
the sale price of the house."""
coef = pd.Series(regressor.feature_importances_, index = X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
coef.head(25).plot(kind='bar')
plt.title('Feature Significance based on Random Forest Regressor')
plt.tight_layout()
plt.show()
kfold = model_selection.ShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
results = model_selection.cross_val_score(regressor, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
test_df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
test_df.drop(['Id', 'GarageYrBlt'], axis=1,inplace=True)
test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna(value='No_FP')
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(value = df['LotFrontage'].mean())
test_df['BsmtCond'] =  test_df['BsmtCond'].fillna(value='No_BS')
test_df['BsmtQual'] =  test_df['BsmtQual'].fillna(value='No_BS')
for col in ('BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    test_df[col] = test_df[col].fillna('No_BS')
for col in ('GarageQual', 'GarageCond', 'GarageFinish','GarageType'):
    test_df[col] = test_df[col].fillna('No_Garage')
test_df.dropna(axis=0, inplace=True)
for col in test_df.columns.values:
       # Encoding only categorical variables
        if test_df[col].dtypes=='object':
            le.fit(test_df[col].values)
            test_df[col]=le.transform(test_df[col])
y_pred = regressor.predict(test_df)
plt.hist(Y, label = 'Train Price')
plt.hist(y_pred, label='Predicted Price')
plt.legend(loc='upper right')
plt.xlabel('Sale Prices')
plt.ylabel('Number of Houses')
plt.title('Trained and predicted prices of the houses')
#plt.text(100000,600,'Average price of train dataset', fontsize=10)
plt.show()
""" Scatter plot showing distribution of original and predicted sale prices w.r.t overall material 
and finish of the house
"""
plt.scatter(df['OverallQual'], Y, marker = '+')
plt.scatter(test_df['OverallQual'], y_pred, marker= 'x')
plt.title('Orginial and predicted sale price w.r.t OverallQual')
plt.show()
