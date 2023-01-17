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
house_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
retain_col = []

for col in house_data.columns:

    if house_data[col].notnull().sum() > 1300:

        retain_col.append(col)
house_data = house_data[retain_col]



feature_data = house_data.drop(columns=['SalePrice'])

target_data = house_data.SalePrice

cat_house_data = feature_data.select_dtypes(include=['object'])
from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder())
float_int_house_data = feature_data.select_dtypes(include=['float','int'])
from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import make_column_transformer



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



from sklearn.model_selection import train_test_split
float_int_pipeline = make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler())
preprocessor = make_column_transformer(

        (cat_pipeline, cat_house_data.columns),

        (float_int_pipeline, float_int_house_data.columns)

)
pipeline = make_pipeline(preprocessor, RandomForestRegressor())


trainX, testX, trainY, testY = train_test_split(feature_data, target_data)
pipeline.fit(feature_data, target_data)
pipeline.score(testX,testY)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

y_pred = pipeline.predict(test)
pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df = pd.DataFrame(y_pred, columns=['SalePrice'])
df.to_csv('output.csv')