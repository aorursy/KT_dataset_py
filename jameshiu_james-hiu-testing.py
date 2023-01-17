# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


home = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')

#home.head()
#test.head()
#Generate feature correlation visualization

plt.figure(figsize=(18,18))
sns.heatmap(home.corr(), annot=True, fmt=".2f", vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', cbar_kws= {'orientation': 'horizontal'} )

home.drop(['MSSubClass','OverallCond','LowQualFinSF','MiscVal','PoolArea','MoSold','YrSold','GarageYrBlt','TotRmsAbvGrd','GarageCars'], axis=1, inplace=True)
test.drop(['MSSubClass','OverallCond','LowQualFinSF','MiscVal','PoolArea','MoSold','YrSold','GarageYrBlt','TotRmsAbvGrd','GarageCars'], axis=1, inplace=True)
#Identify and report count - columns with missing values
missing_data = home.isnull().sum()
col_with_missing = missing_data[missing_data>0]
col_with_missing.sort_values(inplace=True)
print(col_with_missing)

missing_data2 = test.isnull().sum()
col_with_missing2 = missing_data2[missing_data2>0]
col_with_missing2.sort_values(inplace=True)
print(col_with_missing2)

home.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)
test.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)

for df in [home, test]:
#Encode missing categorical features with most common type/quality, grouping by neighborhood
    for col in ("Electrical","MasVnrType", "GarageType","BsmtQual","BsmtCond","BsmtFinType1","BsmtFinType2","BsmtExposure","GarageFinish","GarageQual","GarageCond","FireplaceQu","KitchenQual","SaleType"
               ,"Exterior1st","Exterior2nd","Utilities","Functional","MSZoning"):
         df[col] = df.groupby("Neighborhood")[col].transform(lambda x: x.fillna(x.mode()[0]))
#Encode missing numerical features with average values, grouping by neighborhood
    for col2 in ("MasVnrArea","LotFrontage","TotalBsmtSF","GarageArea","BsmtUnfSF","BsmtFinSF2","BsmtFinSF1","BsmtFullBath","BsmtHalfBath"):
        df[col2] = df.groupby('Neighborhood')[col2].transform(lambda x: x.fillna(x.mean()))
#Identify Numerical Data
numerical_data = home.select_dtypes(exclude=['object']).drop('SalePrice', axis=1)

#Explore Numerical Data Distribution
fig = plt.figure(figsize=(20,18))
for i in range(len(numerical_data.columns)):
    fig.add_subplot (9,4,i+1)
    sns.distplot(a=numerical_data.iloc[:,i].dropna(), kde=False)
    plt.xlabel(numerical_data.columns[i])
plt.tight_layout()
plt.show()
#Identify Categorical Data
categorical_data = home.select_dtypes(['object'])

#Explore Categorical Data Distribution
fig = plt.figure(figsize=(20,18))
for i in range(len(categorical_data.columns)):
    fig.add_subplot(12,4,i+1)
    sns.countplot(x=categorical_data.iloc[:,i])
plt.tight_layout()
plt.show()

#Fit the Model on Training Data
y = home.SalePrice
X = home.drop(['SalePrice'],axis=1)
# Get list of categorical variables
a = (home.dtypes == 'object')
object_cols = list(a[a].index)
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(home[object_cols]))
# One-hot encoding removed index; put it back
OH_cols_train.index = home.index
# Remove categorical columns (will replace with one-hot encoding)
X = X.drop(object_cols, axis=1)

my_model = XGBRegressor()
my_model.fit(X, y)

predictions = my_model.predict(X)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y)))


#Apply the model to Test Data
X_test = test
# Get list of categorical variables
b = (test.dtypes == 'object')
object_cols2 = list(b[b].index)
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_test = pd.DataFrame(OH_encoder.fit_transform(test[object_cols2]))
# One-hot encoding removed index; put it back
OH_cols_test.index = test.index
# Remove categorical columns (will replace with one-hot encoding)
X_test = test.drop(object_cols, axis=1)


predictions2 = my_model.predict(X_test)

print(predictions2)
# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions2})
output.to_csv('submission.csv', index=False)