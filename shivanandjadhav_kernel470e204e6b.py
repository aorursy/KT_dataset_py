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

housing=pd.read_csv('../input/shivhou/shivhou.csv')

housing.head(5)
housing.info()
housing['CHAS'].value_counts() # in this data value one or zero caluculet in CHAS col

                               # this is acatgericl veliabal
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))
from sklearn.model_selection import train_test_split # the shuffle and split is usid in sklearn model.

train_set ,test_set = train_test_split(housing, test_size= 0.2, random_state=43)

print(f"Row in train set: {len(train_set)}\Row in test set:{len(test_set)}\n")
from sklearn.model_selection import StratifiedShuffleSplit 

split = StratifiedShuffleSplit(n_splits= 1 , test_size=0.2 , random_state=43)

for train_index,test_index in split.split(housing,housing['CHAS']):

    strat_train_set =housing.loc[train_index]

    strat_test_set =housing.loc[test_index]

    
strat_test_set['CHAS'].value_counts()
strat_train_set['CHAS'].value_counts()
housing =strat_train_set.copy()
corr_matrix = housing.corr()

corr_matrix ['MEDV'].sort_values(ascending = False)# colearetion is 1 to -1 

                                                    # 1 is the stroing     
from pandas.plotting import scatter_matrix

attributes = ["MEDV","RM", "ZN","LSTAT"]

scatter_matrix(housing[attributes],figsize=(12,8))
housing.plot(kind= "scatter", x ="RM", y="MEDV",alpha=(0.8))
housing["TAXRM"] = housing["TAX"]/housing["RM"]

housing.head()

# new attribute set

corr_matrix = housing.corr()

corr_matrix ['MEDV'].sort_values(ascending = False)
housing.plot(kind= "scatter", x ="TAXRM", y="MEDV",alpha=(0.8))
housing= strat_train_set.drop('MEDV', axis=1)

housing_labels = strat_train_set['MEDV'].copy()
a = housing.dropna(subset=["RM"])  # potion 1

a.shape  #Get rid of the missing data points  
housing.drop("RM",axis=1).shape   # Note there is no RM collmn

#Get rid of the Whole attribute # option 2
median = housing["RM"].median() # comput median for option 

median
housing["RM"].fillna(median) #Set the value to same value (0,mean or mediam)
housing.shape
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")

imputer.fit(housing)
imputer.statistics_
x = imputer.transform(housing)

housing_tr =pd.DataFrame(x,columns=housing.columns)

housing_tr.describe()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([

    ('inputer', SimpleImputer(strategy="median")),

    ('std_scaler',StandardScaler()),

                      ])

housing_num_tr = my_pipeline.fit_transform(housing_tr)

housing_num_tr = my_pipeline.fit_transform(housing)

housing_num_tr 

housing_num_tr.shape

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

#model= DecisionTreeRegressor()

#model = LinearRegression()

model = RandomForestRegressor()

model.fit(housing_num_tr , housing_labels)
some_data =housing .iloc[:5]
some_datals =housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_datals)
from sklearn.metrics import mean_squared_error

housing_predictctions = model.predict(housing_num_tr)

mse = mean_squared_error(housing_labels, housing_predictctions)

rmse = np.sqrt(mse)
rmse
mse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, housing_num_tr,housing_labels , scoring ="neg_mean_squared_error",cv=10)

rmse_scores = np.sqrt(-scores)
rmse_scores
def print_scores(scores):

    print("scores:",scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
print_scores(rmse_scores)
from joblib import dump,load

dump(model, 'Shiv.joblib')
x_test =strat_test_set.drop('MEDV', axis=1)

y_test =strat_test_set["MEDV"].copy()

x_test_prepared = my_pipeline.transform(x_test)

final_predictions= model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

final_rnse = np.sqrt(final_mse)

print(final_predictions, list(y_test))
final_rnse
prepared_data[0]
from joblib import dump,load

import numpy as np

model= load( 'Shiv.joblib')
features= np.array([[0.4048055 ,  -0.88894949, -1.3932448 , -0.27288841, -1.29089824,

        1.40244746, -1.24839122,  0.65332037, 20.85275281, 0.44492981,

       -2.60558671,  10.42866779, -8.19447262]])

model.predict(features)