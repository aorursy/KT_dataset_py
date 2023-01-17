import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
import os

print(os.listdir("../input"))

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
sale_price = train_df["SalePrice"]
test_df.head()
train_df.shape, test_df.shape
train_df.info()
test_df.info()
train_df.describe(include="all") # include both categorical and numerical features
train_df.drop("Id", axis =1,inplace = True)

test_id = test_df["Id"]

test_df.drop("Id", axis =1,inplace = True)
train_df.shape,test_df.shape
from scipy import stats
plt.subplots(figsize=(12,9))

sns.distplot(train_df['SalePrice'], fit=stats.norm)
train_df['SalePrice'] = np.log1p(train_df['SalePrice']) # log transform to make it more "normal"
plt.subplots(figsize=(12,9))

sns.distplot(train_df['SalePrice'], fit=stats.norm)
all_df = pd.concat((train_df,test_df)).reset_index(drop = True)
all_df.drop(['SalePrice'],axis =1,inplace = True)
all_df.shape
numeric_cols = list(all_df._get_numeric_data().columns) 
print((numeric_cols))
numeric_cols_df = all_df[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 

                       'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

                       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',

                       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 

                       'WoodDeckSF', 

                       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 

                       'YrSold']]
numeric_cols_df.shape
numeric_cols_df.hist(bins =20, figsize = (20,20))

plt.show()
for i in train_df[numeric_cols].columns:

    plt.figure(figsize = (8,4))

    plt.scatter( train_df[numeric_cols][i],train_df["SalePrice"])

    plt.xlabel(i)

    plt.ylabel("SalePrice")
corr = train_df[train_df._get_numeric_data().columns].corr()

plt.subplots(figsize=(20,9))

sns.heatmap(corr)
# filtering out only highly co-related features

top_feature = corr.index[abs(corr['SalePrice']>0.5)]

plt.subplots(figsize=(12, 8))

top_corr = train_df[train_df._get_numeric_data().columns][top_feature].corr()

sns.heatmap(top_corr, annot=True)

plt.show()
train_df["OverallQual"].unique() # listing values for highest correlated feature with target variable
sns.barplot(train_df.OverallQual, train_df.SalePrice) # plotting the highest correlated feature
plt.figure(figsize=(18, 8))

sns.boxplot(x=train_df.OverallQual, y=train_df.SalePrice)
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',

       '1stFlrSF']

sns.set(style='ticks')

sns.pairplot(train_df[col], height=3, kind='reg')
data_nan = (numeric_cols_df.isnull().sum() / len(numeric_cols_df)) * 100

data_nan = data_nan.drop(data_nan[data_nan == 0].index).sort_values(ascending=False)[:20]

missing_data = pd.DataFrame({'Missing Ratio' :data_nan})

missing_data
plt.figure(figsize = (8,5))

sns.barplot(x=data_nan.index, y=data_nan)

plt.xlabel('Features', fontsize=15)

plt.xticks(rotation= 'vertical' )

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
all_df["LotFrontage"] = all_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
all_df["MasVnrArea"] = all_df["MasVnrArea"].fillna(0)
all_df["GarageYrBlt"] = all_df["GarageYrBlt"].fillna(0)
all_df["BsmtFinSF2"] = all_df["BsmtFinSF2"].fillna(0)
all_df['BsmtFinSF1'] = all_df["BsmtFinSF1"].fillna(0)
all_df["BsmtHalfBath"] = all_df["BsmtHalfBath"].fillna(0)
all_df['BsmtFullBath'] = all_df["BsmtFullBath"].fillna(0)
all_df['GarageArea'] = all_df["GarageArea"].fillna(0)
all_df["GarageCars"] = all_df["GarageCars"].fillna(0)
all_df["TotalBsmtSF"] = all_df["TotalBsmtSF"].fillna(0)
all_df["BsmtUnfSF"] = all_df["BsmtUnfSF"].fillna(0)
categ_col = list(set(all_df.columns.unique()) - set(numeric_cols))
all_df.shape
print(len(categ_col))
print((categ_col))
categ_col_df = all_df[['Neighborhood', 'MSZoning', 'RoofStyle', 'SaleCondition', 'HouseStyle', 'Utilities', 'LandContour',

                        'MasVnrType', 'Functional', 'Condition1', 'KitchenQual', 'ExterQual', 'PoolQC', 'Foundation',

                        'Heating', 'LotConfig', 'GarageCond', 'LandSlope', 'Street', 'Exterior2nd', 'BsmtQual', 

                        'Exterior1st', 'GarageFinish', 'BsmtExposure', 'GarageType', 'HeatingQC', 'CentralAir', 

                        'PavedDrive', 'SaleType', 'BsmtCond', 'RoofMatl', 'Alley', 'LotShape', 'BldgType', 'BsmtFinType1', 

                        'GarageQual', 'Electrical', 'Fence', 'MiscFeature', 'ExterCond', 'FireplaceQu', 'BsmtFinType2',

                        'Condition2']]
for i in categ_col_df:

    train_df.boxplot("SalePrice",i, rot = 30, figsize = (12,6))
data_nan1 = (categ_col_df.isnull().sum() / len(categ_col_df)) * 100

data_nan1 = data_nan1.drop(data_nan1[data_nan1 == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :data_nan1})

missing_data
plt.figure(figsize = (8,5))

plt.xticks(rotation =90)

sns.barplot(x=data_nan1.index, y=data_nan1)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
all_df["PoolQC"] = all_df["PoolQC"].fillna("None")
all_df["MiscFeature"] = all_df["MiscFeature"].fillna("None")
all_df["Alley"] = all_df["Alley"].fillna("None")
all_df["Fence"] = all_df["Fence"].fillna("None")
all_df["FireplaceQu"] = all_df["FireplaceQu"].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_df[col] = all_df[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_df[col] = all_df[col].fillna('None')
all_df["MasVnrType"] = all_df["MasVnrType"].fillna("None")
all_df["MSZoning"] = all_df["MSZoning"].fillna("None")
all_df["Exterior1st"] =all_df["Exterior1st"].fillna("None")
all_df["Exterior2nd"] = all_df["Exterior2nd"].fillna("None")
all_df["Functional"] = all_df["Functional"].fillna("None")
all_df["SaleType"] = all_df["SaleType"].fillna("None")
all_df["KitchenQual"] = all_df["KitchenQual"].fillna("None")
all_df['Electrical'] = all_df['Electrical'].fillna(all_df['Electrical'].mode()[0])
all_df = all_df.drop(['Utilities'], axis=1)
all_df.isnull().values.sum()
from sklearn.preprocessing import LabelEncoder
cols = ('Neighborhood', 'MSZoning', 'RoofStyle', 'SaleCondition', 'HouseStyle', 'LandContour',

                        'MasVnrType', 'Functional', 'Condition1', 'KitchenQual', 'ExterQual', 'PoolQC', 'Foundation',

                        'Heating', 'LotConfig', 'GarageCond', 'LandSlope', 'Street', 'Exterior2nd', 'BsmtQual', 

                        'Exterior1st', 'GarageFinish', 'BsmtExposure', 'GarageType', 'HeatingQC', 'CentralAir', 

                        'PavedDrive', 'SaleType', 'BsmtCond', 'RoofMatl', 'Alley', 'LotShape', 'BldgType', 'BsmtFinType1', 

                        'GarageQual', 'Electrical', 'Fence', 'MiscFeature', 'ExterCond', 'FireplaceQu', 'BsmtFinType2',

                        'Condition2')
for c in cols:

    label = LabelEncoder() 

    label.fit(list(all_df[c].values)) 

    all_df[c] = label.transform(list(all_df[c].values))
all_df.shape
all_df.head()
ntrain = train_df.shape[0]

ntrain
ntest = test_df.shape[0]

ntest
new_train_df = all_df[:ntrain].copy()
new_train_df["SalePrice"] = sale_price
new_train_df['SalePrice'].head()
test_df = all_df[:ntest]

test_df.shape
X = new_train_df.drop("SalePrice", axis = 1)
Y = new_train_df["SalePrice"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Accuracy is", model.score(X_test, y_test)*100)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=1000)

model.fit(X_train, y_train)

print("Accuracy is ", model.score(X_test, y_test)*100)
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)

GBR.fit(X_train, y_train)

print("Accuracy is ", GBR.score(X_test, y_test)*100)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

ridge = Ridge()

MSEs = cross_val_score(ridge, X, Y, cv=5)

print(MSEs)

mean_MSE = np.mean(MSEs)

print(mean_MSE)
#Test predictions
predicted_price = pd.Series(GBR.predict(test_df), name = "SalePrice")



submission_df = pd.concat([test_id,predicted_price], axis=1)
submission_df.head()
# filename = 'House SalePrice Predictions 1.csv'



# submission_df.to_csv(filename,index=False)



# print('Saved file: ' + filename)