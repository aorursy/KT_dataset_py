# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install sklearn_pandas==2.0.0
trainDF = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
testDF = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
cols_to_ignore = ['Id', 'SalePrice']
for c in trainDF.columns:
    if trainDF[c].isnull().sum() / trainDF.shape[0] > 0.5:
        cols_to_ignore.append(c)

print(cols_to_ignore)

cols_to_consider = set(trainDF.columns) - set(cols_to_ignore)
print(cols_to_consider)
dtype = trainDF[cols_to_consider].dtypes.rename('DataType').to_frame()
uniqVals = trainDF[cols_to_consider].apply(lambda x: x.nunique()).rename('NumUniqueValues')
for k in pd.concat([dtype, uniqVals], ignore_index=False, axis=1).reset_index().to_dict(orient='record'):
    print(k)


from sklearn_pandas import DataFrameMapper, gen_features, NumericalTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
categorical_variables = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
                         'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
                         'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                         'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                         'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 
                         'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                         'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
numerical_variables = [
    'LotFrontage', 'YrSold', 'MoSold', 'OverallCond', 
    'LotArea', 'MasVnrArea', 'GrLivArea', 'GarageArea', 'PoolArea',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF', 'BsmtFinSF1', 'BsmtFinSF2'
]


transformer = DataFrameMapper(
    gen_features([[_] for _ in categorical_variables], classes=[
        {'class': SimpleImputer, 'strategy': 'most_frequent'},
        {'class': OneHotEncoder, 'sparse': False}
    ])
    + 
    gen_features([[_] for _ in numerical_variables], classes=[
        {'class': SimpleImputer, 'strategy': 'mean'},
        {'class': NumericalTransformer, 'func': 'log1p'}
    ])
    + 
    [
        (['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], FunctionTransformer(lambda x: np.log1p(np.nansum(x, axis=1))), {"alias": 'TotalArea'})
    ]
    
    + 
    gen_features(
        [[_] for _ in categorical_variables + numerical_variables],
        [
            {'class': MissingIndicator, 'features': 'all', 'sparse': False, 'error_on_new': False }
        ],
        suffix='_na'    
    )
    , df_out=True
)
trainF = transformer.fit_transform(trainDF)

trainF.sample(10)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor().fit(trainF, np.log1p(trainDF['SalePrice']))
testF = transformer.transform(testDF)
testDF['SalePrice'] = np.exp(model.predict(testF))-1
print('Id,SalePrice')
for row in testDF[['Id', 'SalePrice']].to_dict(orient='records'):
    print(f"{row['Id']},{row['SalePrice']}")
np.array([
    [1, 1, np.nan],
    [2, 2, 2]
]).nansum(axis=1)
from sklearn.model_selection import train_test_split
t1DF, t2DF = train_test_split(trainDF, test_size=0.1)
print(t1DF.shape)
print(t2DF.shape)
t1F = transformer.fit_transform(t1DF)
model = RandomForestRegressor().fit(t1F, np.log1p(t1DF['SalePrice']))

t2DF['Predicted'] = model.predict(transformer.transform(t2DF))
mean_squared_error(np.log1p(t2DF['SalePrice']), t2DF['Predicted'])
testDF['SalePrice'] = np.exp(model.predict(transformer.transform(testDF))) - 1
for i in sorted(zip(transformer.transformed_names_, model.feature_importances_), key=lambda x: abs(x[1]), reverse=True):
    print(i)
