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



df= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df.head()
df.info()
y = df.SalePrice

y
#Dropping the columns with too many missing values

df1= df.drop(columns={'MiscFeature' , 'Fence' , 'PoolQC' , 'Alley' , 'SalePrice'} , axis=1)

df1.info()
#Filling the Null values with most frequent values

import numpy as np

from sklearn.impute import SimpleImputer



def fit_missing_values(column):

  imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

  imputer = imputer.fit(df1[[column]])

  df1[column] = imputer.transform(df1[[column]])
col_missing_values=df1.columns[df1.isnull().any()].tolist()



for i in col_missing_values:

   fit_missing_values(i)
df1.isnull().sum().sum()
#Scaling down the values of all numerical columns

num_cols = df1.columns[df1.dtypes.apply(lambda c: np.issubdtype(c, np.int64))]

print(len(num_cols))



float_cols = df1.columns[df1.dtypes.apply(lambda c: np.issubdtype(c, np.float64))]

print(len(float_cols))
from sklearn.preprocessing import MinMaxScaler



std_num = MinMaxScaler()

df1[num_cols] = std_num.fit_transform(df1[num_cols])



std_float = MinMaxScaler()

df1[float_cols] = std_float.fit_transform(df1[float_cols])
df1.head()
#OneHotEncoding the Categorical Columns

df2 = pd.get_dummies(df1)

df2.info()
df2.columns
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_test.head()
df_test.info()
#Creating the Submission Dataframe

sub = pd.DataFrame(columns=['Id','SalePrice'])



sub['Id'] = df_test['Id'].astype(int)
df_test1= df_test.drop(columns={'MiscFeature' , 'Fence' , 'PoolQC' , 'Alley' } , axis=1)

df_test1.info()


import numpy as np

from sklearn.impute import SimpleImputer



def fit_missing_values_t(column):

  imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

  imputer = imputer.fit(df_test1[[column]])

  df_test1[column] = imputer.transform(df_test1[[column]])


col_missing_values_test = df_test1.columns[df_test1.isnull().any()].tolist()

for i in col_missing_values_test:

   fit_missing_values_t(i)
df_test1.isnull().sum().sum()
num_col_test = df_test1.columns[df_test1.dtypes.apply(lambda c: np.issubdtype(c, np.int64))]

print(len(num_col_test))



float_col_test = df_test1.columns[df_test1.dtypes.apply(lambda c: np.issubdtype(c, np.float64))]

print(len(float_col_test))
from sklearn.preprocessing import MinMaxScaler



std_num = MinMaxScaler()

df_test1[num_col_test] = std_num.fit_transform(df_test1[num_col_test])



float_num = MinMaxScaler()

df_test1[float_col_test] = float_num.fit_transform(df_test1[float_col_test])



df_test1.info() 
df_test2 = pd.get_dummies(df_test1)

df_test2.info()
X_test = df_test2
not_common=list(set(df2.columns).difference(set(X_test.columns)))

len(not_common)
X = df2.drop(columns=not_common , axis=1)

len(X.columns)
from sklearn.ensemble import RandomForestRegressor





rfr = RandomForestRegressor(criterion = 'mse' , n_estimators = 100 , max_depth=12 , min_samples_split=3 , min_samples_leaf=3)



model = rfr.fit(X, y)

print('train accuracy',rfr.score(X, y))
y_pred = model.predict(X_test)
sub['SalePrice'] = y_pred
sub.to_csv('sub.csv', index=False) 