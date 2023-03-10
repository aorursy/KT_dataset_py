import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from datetime import date
from sklearn.ensemble import RandomForestClassifier
pd.options.display.max_columns = None
pd.options.display.max_rows = None
%matplotlib inline
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.info()
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()
df_test.info()
lot_frontage_mean = df['LotFrontage'].mean()
df['LotFrontage'].fillna(lot_frontage_mean, inplace = True)
print('Replaced the missing values for the LotFrontage Column with it\'s Mean: ' + str(lot_frontage_mean))
test_lot_frontage_mean = df_test['LotFrontage'].mean()
df_test['LotFrontage'].fillna(test_lot_frontage_mean, inplace = True)
print('Replaced the missing values in the Test Set for the LotFrontage Column with it\'s Mean: ' + str(test_lot_frontage_mean))
print(df['Alley'].unique())
print(df_test['Alley'].unique())

# df.drop('Alley', axis = 1, inplace = True)
# df_test.drop('Alley', axis = 1, inplace = True)
df['Alley'].fillna('None', inplace = True)
df_test['Alley'].fillna('None', inplace = True)
df['MasVnrArea'].fillna(0, inplace = True)
df_test['MasVnrArea'].fillna(0, inplace = True)
print('Replaced the missing values for the MasVnrArea Column with 0')
df['MasVnrType'].fillna('None', inplace = True)
df_test['MasVnrType'].fillna('None', inplace = True)

print('Replaced the missing values for the MasVnrArea Column with None')
print(df['MasVnrType'].unique())
print(df_test['MasVnrType'].unique())
print(df['FireplaceQu'].unique())
print(df_test['FireplaceQu'].unique())

# df.drop('FireplaceQu', axis = 1, inplace = True)
# df_test.drop('FireplaceQu', axis = 1, inplace = True)

df['FireplaceQu'].fillna('None', inplace = True)
df_test['FireplaceQu'].fillna('None', inplace = True)
print(df['GarageType'].unique())
print(df['GarageFinish'].unique())
print(df['GarageQual'].unique())
print(df['GarageCond'].unique())

df['GarageType'].fillna('None', inplace = True)
df['GarageFinish'].fillna('None', inplace = True)
df['GarageQual'].fillna('None', inplace = True)
df['GarageCond'].fillna('None', inplace = True)
print(df_test['GarageType'].unique())
print(df_test['GarageFinish'].unique())
print(df_test['GarageQual'].unique())
print(df_test['GarageCond'].unique())

df_test['GarageType'].fillna('None', inplace = True)
df_test['GarageFinish'].fillna('None', inplace = True)
df_test['GarageQual'].fillna('None', inplace = True)
df_test['GarageCond'].fillna('None', inplace = True)
print(df['BsmtQual'].unique())
print(df['BsmtCond'].unique())
print(df['BsmtExposure'].unique())
print(df['BsmtFinType1'].unique())
print(df['BsmtFinType2'].unique())

df['BsmtQual'].fillna('None', inplace = True)
df['BsmtCond'].fillna('None', inplace = True)
df['BsmtExposure'].fillna('None', inplace = True)
df['BsmtFinType1'].fillna('None', inplace = True)
df['BsmtFinType2'].fillna('None', inplace = True)
print(df_test['BsmtQual'].unique())
print(df_test['BsmtCond'].unique())
print(df_test['BsmtExposure'].unique())
print(df_test['BsmtFinType1'].unique())
print(df_test['BsmtFinType2'].unique())

df_test['BsmtQual'].fillna('None', inplace = True)
df_test['BsmtCond'].fillna('None', inplace = True)
df_test['BsmtExposure'].fillna('None', inplace = True)
df_test['BsmtFinType1'].fillna('None', inplace = True)
df_test['BsmtFinType2'].fillna('None', inplace = True)
# print(df['BsmtFinSF1'].unique())
# print(df['BsmtFinSF2'].unique())
# print(df['BsmtUnfSF'].unique())
# print(df['TotalBsmtSF'].unique())
# print(df['BsmtFullBath'].unique())
# print(df['BsmtHalfBath'].unique())

df['BsmtFinSF1'].fillna(0, inplace = True)
df['BsmtFinSF2'].fillna(0, inplace = True)
df['BsmtUnfSF'].fillna(0, inplace = True)
df['TotalBsmtSF'].fillna(0, inplace = True)
df['BsmtFullBath'].fillna(0, inplace = True)
df['BsmtHalfBath'].fillna(0, inplace = True)
# print(df_test['BsmtFinSF1'].unique())
# print(df_test['BsmtFinSF2'].unique())
# print(df_test['BsmtUnfSF'].unique())
# print(df_test['TotalBsmtSF'].unique())
# print(df_test['BsmtFullBath'].unique())
# print(df_test['BsmtHalfBath'].unique())

df_test['BsmtFinSF1'].fillna(0, inplace = True)
df_test['BsmtFinSF2'].fillna(0, inplace = True)
df_test['BsmtUnfSF'].fillna(0, inplace = True)
df_test['TotalBsmtSF'].fillna(0, inplace = True)
df_test['BsmtFullBath'].fillna(0, inplace = True)
df_test['BsmtHalfBath'].fillna(0, inplace = True)
print(df['GarageYrBlt'].unique())
GarageYrBlt = df['GarageYrBlt'].dropna().median()
print('Replacing Missing Values for the Garage Year Built Column with it\'s Median ' + str(GarageYrBlt))

df['GarageYrBlt'].fillna(GarageYrBlt, inplace = True)
print(df_test['GarageYrBlt'].unique())
GarageYrBlt = df_test['GarageYrBlt'].dropna().median()
print('Replacing Missing Values for the Garage Year Built Column with it\'s Median ' + str(GarageYrBlt))

df_test['GarageYrBlt'].fillna(GarageYrBlt, inplace = True)
print(df['GarageArea'].unique())
print(df['GarageQual'].unique())

df['GarageArea'].fillna(0, inplace = True)
df['GarageQual'].fillna(0, inplace = True)
print(df_test['GarageArea'].unique())
print(df_test['GarageQual'].unique())

df_test['GarageArea'].fillna(0, inplace = True)
df_test['GarageQual'].fillna(0, inplace = True)
print(df['MiscFeature'].unique())
print(df_test['MiscFeature'].unique())

# df.drop('MiscFeature', axis = 1, inplace = True)
# df_test.drop('MiscFeature', axis = 1, inplace = True)

df['MiscFeature'].fillna('None', inplace = True)
df_test['MiscFeature'].fillna('None', inplace = True)
df.drop('Utilities', axis = 1, inplace = True)
df_test.drop('Utilities', axis = 1, inplace = True)
print(df['Fence'].unique())
print(df_test['Fence'].unique())

# df.drop('Fence', axis = 1, inplace = True)
# df_test.drop('Fence', axis = 1, inplace = True)

df['Fence'].fillna('None', inplace = True)
df_test['Fence'].fillna('None', inplace = True)
print(df['PoolQC'].unique())
print(df_test['PoolQC'].unique())

# df.drop('PoolQC', axis = 1, inplace = True)
# df_test.drop('PoolQC', axis = 1, inplace = True)

df['PoolQC'].fillna('None', inplace = True)
df_test['PoolQC'].fillna('None', inplace = True)
df.dropna(inplace = True)
df.info()
print(df_test['MSZoning'].unique())
MSZoning = df.MSZoning.mode()
df_test['MSZoning'].fillna(MSZoning[0], inplace = True)
print(df_test['Exterior1st'].unique())
Exterior1st = df.Exterior1st.mode()
df_test['Exterior1st'].fillna(Exterior1st[0], inplace = True)
print(df_test['Exterior2nd'].unique())
Exterior2nd = df.Exterior2nd.mode()
df_test['Exterior2nd'].fillna(Exterior2nd[0], inplace = True)
print(df_test['KitchenQual'].unique())
KitchenQual = df.KitchenQual.mode()
df_test['KitchenQual'].fillna(KitchenQual[0], inplace = True)
print(df_test['Functional'].unique())
Functional = df.Functional.mode()
df_test['Functional'].fillna(Functional[0], inplace = True)
print(df_test['SaleType'].unique())
SaleType = df.SaleType.mode()
df_test['SaleType'].fillna(SaleType[0], inplace = True)
print(df_test['GarageCars'].unique())
df_test['GarageCars'].fillna('None', inplace = True)
df_test[df_test['GarageCars'] == 'None'].GarageArea 
df_test['GarageCars'].replace(to_replace = ['None'], value = np.nan, inplace=True)
df_test['GarageCars'].fillna(0, inplace = True)
print(df_test['GarageCars'].unique())
#df_test.dropna(inplace = True)
df_test.info()
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(8,5)})

sns.distplot(a = df['SalePrice'], bins = 50, color = 'gray', vertical = False
            ).set_title('Sale Price')
corr = df.corr()
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(30,30)})

sns.heatmap(corr, cmap="YlGnBu")

sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(8, 8)})
sns.scatterplot(x = "SalePrice", y = "LotArea", data = df, color = "gray")
sns.set(rc={'figure.figsize':(13, 8)})
ax = sns.countplot(df['MSZoning'], color = 'gray')
ax.set(xlabel = "Residential Low Density, Residential Medium Density, Commercial, Floating Village Residential, Residential High Density")
sns.set(rc={'figure.figsize':(8, 8)})
sns.scatterplot(x = "SalePrice", y = "LotFrontage", data = df, color = "gray")
sns.set(rc={'figure.figsize':(8, 8)})
sns.countplot(df['Street'], color = 'gray')
sns.set(rc={'figure.figsize':(6, 6)})
ax = sns.countplot(df['LotShape'], color = 'gray')
ax.set(xlabel = "Regular, Slightly irregular, Moderately Irregular, Irregular")
sns.set(rc={'figure.figsize':(8, 8)})
sns.countplot(df['LandContour'], color = 'gray')
sns.set(rc={'figure.figsize':(8, 5)})
sns.countplot(df['LotConfig'], color = 'gray')
sns.countplot(df['LandSlope'], color = 'gray')
sns.set(rc={'figure.figsize':(21, 5)})
sns.countplot(df['Neighborhood'], color = 'gray')
sns.set(rc={'figure.figsize':(8, 5)})
sns.countplot(df['Condition1'], color = 'gray')
sns.set(rc={'figure.figsize':(5, 5)})
sns.countplot(df['BldgType'], color = 'gray')
sns.set(rc={'figure.figsize':(8, 5)})
sns.countplot(df['HouseStyle'], color = 'gray')
sns.countplot(df['OverallQual'], color = 'gray')
sns.countplot(df['OverallCond'], color = 'gray')
sns.scatterplot(x = "SalePrice", y = "YearBuilt", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "YearRemodAdd", data = df, color = "gray")
sns.countplot(df['RoofStyle'], color = 'gray')
sns.countplot(df['RoofMatl'], color = 'gray')
sns.set(rc={'figure.figsize':(12, 5)})
sns.countplot(df['Exterior1st'], color = 'gray')
sns.countplot(df['Exterior2nd'], color = 'gray')
sns.set(rc={'figure.figsize':(8, 5)})
sns.countplot(df['MasVnrType'], color = 'gray')
sns.scatterplot(x = "SalePrice", y = "MasVnrArea", data = df, color = "gray")
sns.countplot(df['ExterQual'], color = 'gray')
sns.countplot(df['ExterCond'], color = 'gray')
sns.countplot(df['Foundation'], color = 'gray')
sns.countplot(df['BsmtQual'], color = 'gray')
sns.countplot(df['BsmtCond'], color = 'gray')
sns.countplot(df['BsmtExposure'], color = 'gray')
sns.countplot(df['BsmtFinType1'], color = 'gray')
sns.scatterplot(x = "SalePrice", y = "BsmtFinSF1", data = df, color = "gray")
sns.countplot(df['BsmtFinType2'], color = 'gray')
sns.scatterplot(x = "SalePrice", y = "BsmtUnfSF", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "TotalBsmtSF", data = df, color = "gray")
sns.countplot(df['Heating'], color = 'gray')
sns.countplot(df['HeatingQC'], color = 'gray')
sns.countplot(df['CentralAir'], color = 'gray')
sns.countplot(df['Electrical'], color = 'gray')
sns.scatterplot(x = "SalePrice", y = "1stFlrSF", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "2ndFlrSF", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "LowQualFinSF", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "GrLivArea", data = df, color = "gray")
sns.countplot(df['BsmtFullBath'], color = 'gray')
sns.countplot(df['BsmtHalfBath'], color = 'gray')
sns.countplot(df['FullBath'], color = 'gray')
sns.countplot(df['HalfBath'], color = 'gray')
sns.countplot(df['KitchenQual'], color = 'gray')
sns.countplot(df['TotRmsAbvGrd'], color = 'gray')
sns.countplot(df['Functional'], color = 'gray')
sns.countplot(df['Fireplaces'], color = 'gray')
sns.countplot(df['GarageType'], color = 'gray')
sns.scatterplot(x = "SalePrice", y = "GarageYrBlt", data = df, color = "gray")
sns.countplot(df['GarageFinish'], color = 'gray')
sns.countplot(df['GarageCars'], color = 'gray')
sns.scatterplot(x = "SalePrice", y = "GarageArea", data = df, color = "gray")
sns.countplot(df['GarageQual'], color = 'gray')
sns.countplot(df['GarageCond'], color = 'gray')
sns.countplot(df['PavedDrive'], color = 'gray')
sns.scatterplot(x = "SalePrice", y = "WoodDeckSF", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "OpenPorchSF", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "EnclosedPorch", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "3SsnPorch", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "ScreenPorch", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "PoolArea", data = df, color = "gray")
sns.scatterplot(x = "SalePrice", y = "MiscVal", data = df, color = "gray")
sns.countplot(df['MoSold'], color = 'gray')
sns.countplot(df['YrSold'], color = 'gray')
sns.countplot(df['SaleType'], color = 'gray')
sns.countplot(df['SaleCondition'], color = 'gray')
df.columns
df['Remodel'] = df['YearBuilt'] != df['YearRemodAdd']
df['Remodel'] = df['Remodel'].astype(int)
df.head()
df_test['Remodel'] = df_test['YearBuilt'] != df_test['YearRemodAdd']
df_test['Remodel'] = df_test['Remodel'].astype(int)
df_test.head()
today = date.today()
def calculate_age(YearBuilt):
    YearBuilt = str(YearBuilt)
    built = datetime.strptime(YearBuilt, "%Y").date()
    return today.year - built.year 
df['HouseAge'] = df['YearBuilt'].apply(calculate_age)
df.head() 
df_test['HouseAge'] = df_test['YearBuilt'].apply(calculate_age)
df_test.head()
df.drop('YearBuilt', axis = 1, inplace = True)
df.drop('YearRemodAdd', axis = 1, inplace = True)

df_test.drop('YearBuilt', axis = 1, inplace = True)
df_test.drop('YearRemodAdd', axis = 1, inplace = True)
df.head()
df.info()
x_train = df[[col for col in df.columns if col not in ['SalePrice', 'Id']]]
y_train = df['SalePrice']
x_train.head()
y_train.head()
x_train = pd.get_dummies(x_train)
x_train.info()
# Intializing the Model
clf = RandomForestClassifier(max_depth=500, random_state=0)
# Train the Model
clf.fit(x_train, y_train)
x_test = df[[col for col in df_test.columns if col not in ['SalePrice', 'Id']]]
x_test = pd.get_dummies(x_test)
x_test.info()
x_test.head()
prices = clf.predict(x_test)
prices
submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.head()
submission.info()
output = pd.DataFrame({'Id': df_test.Id, 'SalePrice': prices})
output.head()
output.info()
#output.to_csv('Submission_RandomForest.csv', index=False)