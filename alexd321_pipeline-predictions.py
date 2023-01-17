# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/sf-employee-compensation/employee-compensation.csv')

df = df.sample(frac=1).reset_index(drop=True) # shuffle data so train and test can have same categorical variables

df = df[:50000]

print(df.shape)

df.head()
df.isna().sum()
df.drop(['Department', 'Job'], axis=1, inplace=True) # 'Department' and 'Job' are basically duplicates for 'Department Code' and 'Job Code'



## aggregators for 'Total Compensation', get near perfect prediction accuracy if include all of them ##

## keep benefit columns only ##

# 'Salaries' + 'Overtime' + 'Other Salaries' = 'Total Salary' 

# 'Retirement' + 'Health and Dental' + 'Other Benefits' = 'Total Benefits'

# 'Total Salary' + 'Total Benefits' = 'Total Compensation'

df.drop(['Salaries', 'Overtime', 'Other Salaries', 'Total Salary', 'Total Benefits'], axis=1, inplace=True)
df.dtypes
ax = sns.boxplot(x="Year Type", y="Total Compensation", data=df)
plt.figure(figsize=(20,8))

ax = sns.boxplot(x="Organization Group", y="Total Compensation", data=df)

plt.tight_layout()
object_cols = [col for col in df.columns if df[col].dtype == object]

num_cols = [col for col in df.columns if df[col].dtype in [float, int]]



num_cols.remove('Total Compensation') # y-variable
object_col_cnts = {col: df[col].nunique() for col in object_cols}

object_col_cnts
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



X_cols = [col for col in df.columns if col != 'Total Compensation'] # remove output variable



X = df[X_cols]

y = df['Total Compensation']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



## Remove cols from X_train and X_test where categorical values not contained in train and test set ##

bad_lbl_cols = [col for col in object_cols if list(set(X_train[col]) - set(X_test[col]))]

bad_lbl_cols
X_cols = [col for col in X_cols if col not in bad_lbl_cols]

X_train = X_train.drop(bad_lbl_cols, axis=1)

X_test = X_test.drop(bad_lbl_cols, axis=1)

object_cols = [x for x in object_cols if x not in bad_lbl_cols]

object_cols
OH_cols = [k for k,v in object_col_cnts.items() if v <= 10] # only OHE variables w/ less than 10 unique values per column

LE_cols = list(set(object_cols) - set(OH_cols))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder 

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



numerical_transformer = SimpleImputer(missing_values=np.nan, strategy='mean')



# two steps: 

# (1) Fill in missing values using SimpleImputer

# (2) One Hot Encode the variables, creating new columns for each unique type

OH_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                                 ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False)) # ignore errors when calling 'transform', sparse=False returns np.array

])



# Same as OH_transformer except using LabelEncoder instead --> high-cardinality columns

LE_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                                 ('lbl_enc', OrdinalEncoder())

])



# Bundle preprocessing for numerical and two categorical groups data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, num_cols),

        ('oh', OH_transformer, OH_cols),

        ('le', LE_transformer, LE_cols),

    ])
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor



# list of models to evaluate

models = {'Random Forest': RandomForestRegressor(n_estimators=100, random_state=0),

          'KNN': KNeighborsRegressor(),

          'Linear Regression': LinearRegression(),

          'Gradient Boost': GradientBoostingRegressor(random_state=0)

         }
from sklearn.metrics import mean_absolute_error



model_mse = {}

model_adj_r2 = {}



for name, model in models.items():

    pipe = Pipeline(steps=[('preprocessor', preprocessor),

                           ('regressor', model)])

    pipe.fit(X_train, y_train)   

    y_pred = pipe.predict(X_test)



    model_adj_r2[name] = round(pipe.score(X_test, y_test), 3)

    model_mse[name] = mean_absolute_error(y_test, y_pred)
model_adj_r2
model_mse
benefit_cols = ['Retirement', 'Health and Dental', 'Other Benefits']

df.drop(benefit_cols, axis=1, inplace=True)



# object cols stay the same

num_cols = [col for col in num_cols if col not in benefit_cols]
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



X_cols = [col for col in df.columns if col not in (benefit_cols + ['Total Compensation'])] # remove output variable



X = df[X_cols]

y = df['Total Compensation']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# still need to check for values in train set not in test set

bad_lbl_cols = [col for col in object_cols if list(set(X_train[col]) - set(X_test[col]))]

bad_lbl_cols
X_cols = [col for col in X_cols if col not in bad_lbl_cols]

X_train = X_train.drop(bad_lbl_cols, axis=1)

X_test = X_test.drop(bad_lbl_cols, axis=1)

object_cols = [x for x in object_cols if x not in bad_lbl_cols]

object_cols
OH_cols = [k for k,v in object_col_cnts.items() if v <= 10] # only OHE variables w/ less than 10 unique values per column

LE_cols = list(set(object_cols) - set(OH_cols))
X_cols
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder 

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



numerical_transformer = SimpleImputer(missing_values=np.nan, strategy='mean')



# two steps: 

# (1) Fill in missing values using SimpleImputer

# (2) One Hot Encode the variables, creating new columns for each unique type

OH_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                                 ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse=False)) # ignore errors when calling 'transform', sparse=False returns np.array

])



# Same as OH_transformer except using LabelEncoder instead --> high-cardinality columns

LE_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),

                                 ('lbl_enc', OrdinalEncoder())

])



# Bundle preprocessing for numerical and two categorical groups data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, num_cols),

        ('oh', OH_transformer, OH_cols),

        ('le', LE_transformer, LE_cols),

    ])
from sklearn.metrics import mean_absolute_error



model_mse = {}

model_adj_r2 = {}



for name, model in models.items():

    pipe = Pipeline(steps=[('preprocessor', preprocessor),

                           ('regressor', model)])

    pipe.fit(X_train, y_train)   

    y_pred = pipe.predict(X_test)



    model_adj_r2[name] = round(pipe.score(X_test, y_test), 3)

    model_mse[name] = mean_absolute_error(y_test, y_pred)
model_adj_r2
model_mse