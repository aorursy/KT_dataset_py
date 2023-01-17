# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys

np.set_printoptions(threshold=sys.maxsize)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# Loading dataset
train_df = pd.read_csv("../input/widsdatathon2020/training_v2.csv")
test_df = pd.read_csv("../input/widsdatathon2020/unlabeled.csv")


train_df.head()

display(train_df.nunique)
train_df.isna()
train_df.isna().sum()
train_df.describe()
print(test_df.shape)
test_df.describe()
test_df.isna()
test_df.isna().sum()
train_df['hospital_death'].dtype
test_df['hospital_death'].dtype
def display_columns_properties(df):
    for i, col in enumerate(df.columns.tolist()):
         print('\n ({} {})  Missing: {}  UniqValsSz: {}'.format(i,col, df[col].isnull().sum() ,df[col].unique().size))
    print('\n')
    
display_columns_properties(train_df)
display_columns_properties(test_df)
cat_train_df = train_df.select_dtypes(include='object')
cat_train_df.head()
cat_train_df.info()
cat_test_df = test_df.select_dtypes(include='object')
cat_test_df.head()
cat_test_df.info()
def display_columns_uniqvals(df):
    for i, col in enumerate(df.columns.tolist()):
         print('\n ({} {}) Uniq: {}'.format(i,col, df[col].unique() ))
    print('\n')
display_columns_uniqvals(cat_test_df)
from sklearn.model_selection import train_test_split

# copy the data
train = train_df.copy()

# Select target
y = train['hospital_death']

# To keep things simple, we'll use only numerical predictors
predictors = train.drop(['hospital_death'], axis=1)
X = predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

X_train.shape

X_valid.shape
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train)) #fit_transform is used for calculating the mean from columns and then replacing the missing values
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
display_columns_properties(imputed_X_train)
display_columns_properties(imputed_X_valid)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# Define model; Specify a number for random_state to ensure the same results in each run.
dt_model = DecisionTreeRegressor(random_state=1)

# Fit model using Training data
dt_model.fit(imputed_X_train, y_train)

# get predicted values on validation data
predicted_values = dt_model.predict(imputed_X_valid)

# Find difference
score = mean_absolute_error(y_valid, predicted_values)
print('MAE:', score)
# Find difference
score = mean_absolute_error(y_valid, predicted_values)
print('MAE:', score)
test = test_df.copy()

#Separate target
y_test = test['hospital_death']

# To keep things simple, we will only use numerical predictors
predictors_test = test.drop(['hospital_death'], axis=1)
X_test = predictors_test.select_dtypes(exclude=['object'])

X_test.head()
X_test.shape
# Imputation
my_imputer = SimpleImputer()
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

# Imputation removed column names; put them back
imputed_X_test.columns = X_test.columns
imputed_X_test.head()
# get predictions on test data
preds = dt_model.predict(imputed_X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'encounter_id': imputed_X_test.encounter_id,
                       'hospital_death': preds},dtype=np.int32)
print(output)

