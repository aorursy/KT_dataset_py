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
import pandas as pd
trainDF = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
testDF = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
trainDF.sample(4)
import numpy as np
ignore_columns = []
numerical_columns = []
categorical_columns = []

for col in trainDF.columns:
    if col == 'SalePrice':
        continue
        
    c = trainDF[col]
    dtype = c.dtype
    missing = c.isnull().sum()
    uniq = len(np.unique(c.astype('str')))
    
    tmissing = testDF[col].isnull().sum()
    print(f"{col}: {dtype}, {missing} missing, {uniq} Unique Values, {tmissing} Missing In Test")
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LinearRegression
transformer = DataFrameMapper([
    (['MSSubClass'], OneHotEncoder(handle_unknown='ignore', sparse=False)),
    (['MSZoning'], [
        SimpleImputer(strategy='most_frequent'), 
        OneHotEncoder(handle_unknown='ignore', sparse=False)
    ]),
    
    (['LotFrontage'], SimpleImputer(strategy='mean')),
    (['LotFrontage'], MissingIndicator(), {'alias': 'LotFrontage_na'}),
    
    (['Neighborhood'], OneHotEncoder())

], df_out=True)

featuresDF = transformer.fit_transform(trainDF)
model = LinearRegression(normalize=True).fit(featuresDF, trainDF['SalePrice'])
testDF['SalePrice'] = model.predict(transformer.transform(testDF))
ignore_columns = []
numerical_columns = []
categorical_columns = []

for col in trainDF.columns:
    if col in ('Id', 'SalePrice'):
        ignore_columns.append(col)
        continue
    
    missing = trainDF[col].isnull().sum()
    if missing / trainDF.shape[0] > 0.5:
        ignore_columns.append(col)
    elif trainDF[col].dtype in (np.float_, np.int_):
        numerical_columns.append(col)
    else:
        categorical_columns.append(col)
    
print(ignore_columns)
print(numerical_columns)
print(categorical_columns)
from sklearn_pandas import gen_features
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Lasso

transformer = DataFrameMapper(
    # handle categorical data
    gen_features(
        [[x] for x in categorical_columns],
        classes=[
            {'class': SimpleImputer, 'strategy': 'most_frequent'},
            {'class': OneHotEncoder, 'handle_unknown': 'ignore', 'sparse': False}
        ]
    ) 

    + 
    #  handle numerical data
    gen_features(
        [[x] for x in numerical_columns],
        classes=[
            {'class': SimpleImputer, 'strategy': 'mean'},
        ]
    )
    + 
    [
        (['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], FunctionTransformer(lambda x: np.nansum(x, axis=1)), {'alias': 'totalArea'})
    ]
    , df_out=True
)

f = transformer.fit_transform(trainDF)
model = Lasso(normalize=True).fit(f, trainDF['SalePrice'])
testDF['SalePrice'] = model.predict(transformer.transform(testDF))
testDF[['Id', 'SalePrice']].to_csv('/kaggle/working/submission.csv', index=False)
model.coef_
