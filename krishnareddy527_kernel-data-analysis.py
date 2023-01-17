import warnings

warnings.filterwarnings("ignore")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
melobourne_data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

melobourne_data.describe()
melobourne_data.shape
melobourne_data.columns
melobourne_data.describe()
melobourne_data.dropna(inplace=True)

melobourne_data.shape
y = melobourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melobourne_data[melbourne_features]
X.describe()
import numpy as np

import random

random.seed = 42

np.random.seed = 42
from sklearn.tree import DecisionTreeRegressor

dst = DecisionTreeRegressor(random_state=42)
dst.fit(X,y)
print("Making predictions for the following 5 houses:")

print(X.head())

print("The predictions are")

print(dst.predict(X.head()))
melobourne_data.Price.head()
from sklearn.metrics import mean_absolute_error

predicted = dst.predict(X)

mean_absolute_error(y,predicted)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,test_size =0.2 ,random_state = 42)
model2 = DecisionTreeRegressor(random_state=42)

# Fit model

model2.fit(train_X, train_y)
val_predictions = model2.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
def get_mae(max_depth , train_X, val_X, train_y, val_y):

    clf = DecisionTreeRegressor(max_depth = max_depth , random_state=42)

    clf.fit(train_X , train_y)

    pred = clf.predict(val_X)

    mae = mean_absolute_error(pred , val_y)

    return mae
# compare MAE with differing values of max_leaf_nodes

MAE=[]

for max_depth in range(1,100):

    my_mae = get_mae(max_depth, train_X, val_X, train_y, val_y)

    MAE.append(my_mae)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_depth, my_mae))
min(MAE)
MAE.index(min(MAE))
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(random_state=42 )

forest.fit(train_X, train_y)
forest_preds = forest.predict(val_X)

value_error = mean_absolute_error(val_y , forest_preds)

print(value_error)
data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

target = data.Price

predictors = data.drop(["Price"],axis=1)

numeric_predictors = predictors.select_dtypes(exclude=['object'])
cols_with_missing = [col for col in numeric_predictors.columns if numeric_predictors[col].isnull().any()]

print(cols_with_missing)
dropped_data = numeric_predictors.drop(cols_with_missing,axis=1)

dropped_data.columns
def get_score(train_X, val_X, train_y, val_y):

    clf = RandomForestRegressor(random_state=42)

    clf.fit(train_X , train_y)

    pred = clf.predict(val_X)

    mae = mean_absolute_error(pred , val_y)

    return mae
train_X, val_X, train_y, val_y = train_test_split(dropped_data , target,test_size =0.2 ,random_state = 42)
print("Mean Absolute Error from dropping columns with Missing Values:")

print(get_score(train_X, val_X, train_y, val_y))
from sklearn.impute import SimpleImputer

myimpute=SimpleImputer()

impute_train = myimpute.fit_transform(train_X)

impute_val = myimpute.fit_transform(val_X)
print("Mean Absolute Error from Imputation:")

print(get_score(impute_train, impute_val, train_y, val_y))
new_data = numeric_predictors.copy()

print(cols_with_missing)
for col in cols_with_missing:

    new_data[col + '_was_missing'] = new_data[col].isnull()

imputed_data = myimpute.fit_transform(new_data)
impute_extra_train, impute_extra_val, train_y, val_y = train_test_split(imputed_data , target,test_size =0.2 ,random_state = 42)
print("Mean Absolute Error from Imputation while Track What Was Imputed:")

print(get_score(impute_extra_train, impute_extra_val, train_y, val_y))
import os

os.listdir("../input/house-prices-advanced-regression-techniques")
import pandas as pd

Raw_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv")

Raw_data.head()
Raw_data.shape
Raw_data.dropna(axis=0,subset=['SalePrice'],inplace=True)

Raw_data.shape
target = Raw_data.SalePrice
cols_with_missing = [col for col in Raw_data.columns if Raw_data[col].isnull().any()]  

print(cols_with_missing)
Data = Raw_data.drop(['Id', 'SalePrice'] + cols_with_missing,axis = 1)

Data.shape
Data.info()
neumeric_cols = [cname for cname in Data.columns if Data[cname].dtype in ['int64', 'float64']]

len(neumeric_cols)
low_cardinality_cols = [cname for cname in Data.columns if Data[cname].dtype == "object" and Data[cname].nunique()<10]

len(low_cardinality_cols)
cols = neumeric_cols + low_cardinality_cols

new_data = Data[cols]
new_data.shape
new_data.dtypes.sample(10)
one_hot_encoded_data = pd.get_dummies(new_data)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
def get_mae(X, y):

    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention

    return -1 * cross_val_score(RandomForestRegressor(random_state = 42), 

                                X, y, scoring = 'neg_mean_absolute_error').mean()
excluded_data = new_data.select_dtypes(exclude=['object'])

excluded_data.shape
import warnings

warnings.filterwarnings("ignore")
mae_one_hot_encoded  = get_mae(one_hot_encoded_data, target)

mae_not_encoded= get_mae(excluded_data, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_not_encoded)))

print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer
Raw_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv")

Raw_data.head()
y = Raw_data.SalePrice

print(Raw_data.shape)

x=Raw_data.drop("SalePrice",axis=1).select_dtypes(exclude=["object"])

print(x.shape)
train_X, test_X, train_y, test_y = train_test_split(x.as_matrix(), y.as_matrix(), test_size=0.25)
my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators = 1000 ,random_state= 42 , learning_rate = 0.05)

my_model.fit(train_X,train_y,early_stopping_rounds = 5,eval_set=[(test_X, test_y)],verbose=False)
pred = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error

print("mean absolute error "+str(mean_absolute_error(pred,test_y)))
from sklearn.ensemble.partial_dependence import partial_dependence , plot_partial_dependence

from sklearn.ensemble import GradientBoostingRegressor
Raw_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv")

y = Raw_data.SalePrice

x=Raw_data.drop("SalePrice",axis=1).select_dtypes(exclude=["object"])
x.head(n=1)
my_imputer = Imputer()

train_X = my_imputer.fit_transform(x[['YearBuilt', 'LotFrontage', 'OverallCond']])
my_model = GradientBoostingRegressor()

my_model.fit(train_X,y)
my_plots=plot_partial_dependence(my_model,       

                                   features=[0,1, 2], # column numbers of plots we want to show

                                   X=train_X,            # raw predictors data.

                                   feature_names=['YearBuilt', 'LotFrontage', 'OverallCond'], # labels on graphs

                                   grid_resolution=10)
import os

os.listdir("../input/melbourne-housing-snapshot")
data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

y = data.Price

data = data.drop("Price",axis=1).select_dtypes(exclude="object")

features = data.columns


clf =  GradientBoostingRegressor()

mu_impute = Imputer()

imputed_data = mu_impute.fit_transform(data)

clf.fit(imputed_data,y)
#my_plots

my_plots= plot_partial_dependence(clf,features=[0,1,2] ,X=imputed_data ,feature_names= features,grid_resolution=10)
data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

y = data.Price

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

X = data[cols_to_use]
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')

print(scores.mean())