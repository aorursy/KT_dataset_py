import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Read the train and test data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',index_col='Id')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv',index_col='Id')
train.describe()
train.dtypes
train.head()
train.tail()
train.shape,test.shape
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.figure(figsize=(10,6))

plt.title("Price on Year Sold")



sns.barplot(x=train['YrSold'],y=train['SalePrice'])
plt.figure(figsize=(10,6))

plt.title('Price Based On OverallQuality Of House')



sns.barplot(x=train['OverallQual'],y=train['SalePrice'])
plt.figure(figsize=(10,6))

plt.title('Central AC Available Houses')



sns.swarmplot(x=train['CentralAir'],y=train['SalePrice'])

print(train['CentralAir'].value_counts())
plt.figure(figsize=(10,6))

plt.title('Price Based On Exteriror Quality')



sns.barplot(y=train['ExterQual'],x=train['SalePrice'])
plt.figure(figsize=(10,6))

plt.title('Price of House Based on Zone')







sns.violinplot(x=train['MSZoning'], y=train['SalePrice'],data=train)
# Make a copy of train and test data

train_copy = train.copy()

test_copy = test.copy()
# Remove the rows with missing target



train_copy.dropna(axis=0,subset=['SalePrice'],inplace=True)



# separate target column from predictors



y = train_copy.SalePrice



# Remove the target column 'SalePrice' from predictors.



train_copy.drop(['SalePrice'],axis=1,inplace=True)
train_copy.shape
from sklearn.model_selection import train_test_split



# Break off validation set from training data into 80% train data and 20% validation data



X_train_full, X_valid_full, y_train, y_valid =  train_test_split(train_copy, y, train_size=0.8, test_size=0.2, random_state=0)
X_train_full.shape
# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality



low_cardinality_cols = [cname for cname in X_train_full.columns 

                        if X_train_full[cname].nunique()<10 and X_train_full[cname].dtype == "object"]



# select numerical columns



numeric_cols = [cname for cname in X_train_full.columns

                     if X_train_full[cname].dtype in ['int64','float64']]



my_cols = low_cardinality_cols + numeric_cols



X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = test_copy[my_cols].copy()
X_train.shape,X_valid.shape,X_test.shape
y_train.head()
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



# Preprocessing for numerical data



numerical_transformer = Pipeline(steps = [

                        ('imputer', SimpleImputer(strategy='median'))

                        ])



# Preprocessing for categorical data



categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

                                           ('onehot', OneHotEncoder(handle_unknown='ignore'))])



# Bundle preprocessing for categorical and numerical data



preprocessor = ColumnTransformer(

                transformers=[

                  ('num', numerical_transformer, numeric_cols),

                    ('cat', categorical_transformer, low_cardinality_cols)

                ])
# Define the model

from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
clf = Pipeline(steps=[

    ('preprocessor',preprocessor),

    ('model',my_model)

])



# Train the model



clf.fit(X_train,y_train)
# Predicting the SalePrice of X_valid data 



from sklearn.metrics import mean_absolute_error



predictions = clf.predict(X_valid)



# Evaluating the performance 'MAE - Mean Absolute Error'



print("MAE:", mean_absolute_error(y_valid,predictions))
test_predictions = clf.predict(X_test)
# saving the predictions into csv file



output = pd.DataFrame({'Id': X_test.index,

                      'SalePrice': test_predictions})



output.to_csv('submission.csv',index=False)
