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
train_data= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train_data.head()
train_data.info()
#Here low value means drop those columns

train_data_1= train_data.drop(columns={'MiscFeature' , 'Fence' , 'PoolQC' , 'Alley' , 'SalePrice'} , axis=1)

train_data_1.info()
y = train_data.SalePrice

y
#Filling the Null values with most frequent values



from sklearn.impute import SimpleImputer



def fit_missing_values(column):

  imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

  imputer = imputer.fit(train_data_1[[column]])

  train_data_1[column] = imputer.transform(train_data_1[[column]])
col_missing_values=train_data_1.columns[train_data_1.isnull().any()].tolist()



for i in col_missing_values:

   fit_missing_values(i)
#Scaling down the values of all numerical columns

num_cols = train_data_1.columns[train_data_1.dtypes.apply(lambda c: np.issubdtype(c, np.int64))]

print(len(num_cols))



float_cols = train_data_1.columns[train_data_1.dtypes.apply(lambda c: np.issubdtype(c, np.float64))]

print(len(float_cols))
from sklearn.preprocessing import MinMaxScaler



std_num = MinMaxScaler()

train_data_1[num_cols] = std_num.fit_transform(train_data_1[num_cols])



std_float = MinMaxScaler()

train_data_1[float_cols] = std_float.fit_transform(train_data_1[float_cols])
train_data_1.head()
#OneHotEncoding the Categorical Columns

train_data_2 = pd.get_dummies(train_data_1)

train_data_2.info()
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_data.head()
test_data.info()
#Again drop columns with more null values

test_data_1= test_data.drop(columns={'MiscFeature' , 'Fence' , 'PoolQC' , 'Alley' } , axis=1)

test_data_1.info()
def fit_missing_values_t(column):

  imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

  imputer = imputer.fit(test_data_1[[column]])

  test_data_1[column] = imputer.transform(test_data_1[[column]])
col_missing_values_test = test_data_1.columns[test_data_1.isnull().any()].tolist()

for i in col_missing_values_test:

   fit_missing_values_t(i)
num_col_test = test_data_1.columns[test_data_1.dtypes.apply(lambda c: np.issubdtype(c, np.int64))]

print(len(num_col_test))



float_col_test = test_data_1.columns[test_data_1.dtypes.apply(lambda c: np.issubdtype(c, np.float64))]

print(len(float_col_test))
std_num = MinMaxScaler()

test_data_1[num_col_test] = std_num.fit_transform(test_data_1[num_col_test])



float_num = MinMaxScaler()

test_data_1[float_col_test] = float_num.fit_transform(test_data_1[float_col_test])



test_data_1.info() 
test_data_2 = pd.get_dummies(test_data_1)

test_data_2.info()
X_test = test_data_2
not_common=list(set(train_data_2.columns).difference(set(X_test.columns)))

len(not_common)
#Dropping columns that are not common

X = train_data_2.drop(columns=not_common , axis=1)

len(X.columns)
from sklearn.ensemble import RandomForestRegressor



rfr = RandomForestRegressor(criterion = 'mse' , n_estimators = 100 , max_depth=12 , min_samples_split=3 , min_samples_leaf=3, random_state =  1)



model = rfr.fit(X, y)

score_rfr=rfr.score(X, y)

print('train accuracy',rfr.score(X, y))
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(criterion = 'mse' , max_depth=12 , min_samples_split=3 , min_samples_leaf=3)

regressor.fit(X, y)

score_dtr=regressor.score(X, y)

print('train accuracy',regressor.score(X, y))
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X, y)

score_lr=regressor.score(X, y)

print('train accuracy',regressor.score(X, y))
X.shape
y
y.shape
#y = y.reshape(len(y),1)==>'Series' object has no attribute 'reshape'

y = y.values.reshape(len(y),1)
y.shape
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X = sc_X.fit_transform(X)

y = sc_y.fit_transform(y)
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X, y)

score_svr=regressor.score(X, y)

print('train accuracy',regressor.score(X, y))
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(n_estimators=1000, learning_rate=.03, max_depth=3, max_features=.06, min_samples_split=4,

                                 min_samples_leaf=3, loss='huber', subsample=.9, random_state=0)



gbrt.fit(X, y)

score_gbrt=gbrt.score(X,y)
models = pd.DataFrame({'Model': ['Random Forest Regressor','LinearRegression','Decision Tree Regressor','SVR','Gradient Boosting Regressor'],

    'Score': [score_rfr,score_lr,score_dtr,score_svr,score_gbrt]})

models.sort_values(by='Score', ascending=False)