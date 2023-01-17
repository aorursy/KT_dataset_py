import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# A lot area is the total area of a property, including the yard up to the boundaries (property line), while the floor area is the area inside the building that is occupiable, up to and including the exterior walls.

area_train_df = train_df.iloc[:,4:5].values

area_test_df = test_df.iloc[:,4:5].values
price_train_df = train_df.iloc[:,80:].values
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(area_train_df, price_train_df)
# Visualising the Training set results

plt.scatter(area_train_df, price_train_df, color = 'red')

plt.plot(area_train_df, regressor.predict(area_train_df), color = 'blue')

plt.title('Price vs Area (Training set)')

plt.xlabel('Area')

plt.ylabel('Price')

plt.show()
regressor.predict(area_test_df)
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

area_poly = poly_reg.fit_transform(area_train_df)

poly_reg.fit(area_poly, price_train_df)

lin_reg = LinearRegression()

lin_reg.fit(area_poly, price_train_df)
plt.scatter(area_train_df, price_train_df, color = 'red')

plt.plot(area_train_df, lin_reg.predict(poly_reg.fit_transform(area_train_df)), color = 'blue')

plt.title('Price vs Area (Training set)')

plt.xlabel('Area')

plt.ylabel('Price')

plt.show()
lin_reg.predict(poly_reg.fit_transform(area_test_df))
def pre_process_data(df):

    na_columns = [column for column in df.columns if df[column].isnull().values.any()]

    

    # LotFrontage: Linear feet of street connected to property

    df['LotFrontage'] = df['LotFrontage'].fillna(value = df['LotFrontage'].mean())

    # MasVnrArea: Masonry veneer area in square feet

    df['MasVnrArea'] = df['MasVnrArea'].fillna(value = df['MasVnrArea'].mean())

    

    # GarageYrBlt: Year garage was built

    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(value = 0.0)

    

    # Alley: Type of alley access to property

    df['Alley'] = df['Alley'].fillna(value = 'NA')

    # MasVnrType: Masonry veneer type

    df['MasVnrType'] = df['MasVnrType'].fillna(value = 'NA')

    # BsmtQual: Evaluates the height of the basement

    df['BsmtQual'] = df['BsmtQual'].fillna(value = 'NA')

    # BsmtCond: Evaluates the general condition of the basement

    df['BsmtCond'] = df['BsmtCond'].fillna(value = 'NA')

    # BsmtExposure: Refers to walkout or garden level walls

    df['BsmtExposure'] = df['BsmtExposure'].fillna(value = 'NA')

    # BsmtFinType1: Rating of basement finished area

    df['BsmtFinType1'] = df['BsmtFinType1'].fillna(value = 'NA')

    # BsmtFinType2: Rating of basement finished area

    df['BsmtFinType2'] = df['BsmtFinType2'].fillna(value = 'NA')

    # Electrical: Electrical system

    df['Electrical'] = df['Electrical'].fillna(value = 'NA')

    # FireplaceQu: Fireplace quality

    df['FireplaceQu'] = df['FireplaceQu'].fillna(value = 'NA')

    # GarageType: Garage location

    df['GarageType'] = df['GarageType'].fillna(value = 'NA')

    # GarageFinish: Interior finish of the garage

    df['GarageFinish'] = df['GarageFinish'].fillna(value = 'NA')

    # GarageQual: Garage quality

    df['GarageQual'] = df['GarageQual'].fillna(value = 'NA')

    # GarageCond: Garage condition

    df['GarageCond'] = df['GarageCond'].fillna(value = 'NA')

    # PoolQC: Pool quality

    df['PoolQC'] = df['PoolQC'].fillna(value = 'NA')

    # Fence: Fence quality

    df['Fence'] = df['Fence'].fillna(value = 'NA')

    # MiscFeature: Miscellaneous feature not covered in other categories

    df['MiscFeature'] = df['MiscFeature'].fillna(value = 'NA')

    

    # BsmtFinSF1: Type 1 finished square feet

    df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(value = df['BsmtFinSF1'].mean())

    # BsmtFinSF2: Type 2 finished square feet

    df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(value = df['BsmtFinSF2'].mean())

    # BsmtUnfSF: Unfinished square feet of basement area

    df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(value = df['BsmtUnfSF'].mean())

    # TotalBsmtSF: Total square feet of basement area

    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(value = df['TotalBsmtSF'].mean())

    # GarageArea: Size of garage in square feet

    df['GarageArea'] = df['GarageArea'].fillna(value = df['GarageArea'].mean())

    

    # BsmtFullBath: Basement full bathrooms

    df['BsmtFullBath'] = df['BsmtFullBath'].fillna(value = 'NA')

    # BsmtHalfBath: Basement half bathrooms

    df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(value = 'NA')

    # GarageCars: Size of garage in car capacity

    df['GarageCars'] = df['GarageCars'].fillna(value = 'NA')

    

    return df
train_df = pre_process_data(train_df)
cat_cols = [column for column in train_df.columns if train_df[column].dtype == 'object']

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()

for column in cat_cols:

    train_df[column] = label_encoder.fit_transform(train_df[column].astype(str))
area_train_df = train_df.iloc[:,:80].values
from sklearn.linear_model import LinearRegression

multi_regressor = LinearRegression()

multi_regressor.fit(area_train_df, price_train_df)
test_df = pre_process_data(test_df)

cat_cols = [column for column in test_df.columns if test_df[column].dtype == 'object']

for column in cat_cols:

    test_df[column] = label_encoder.fit_transform(test_df[column].astype(str))
multi_regressor.predict(test_df)