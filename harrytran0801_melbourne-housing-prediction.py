# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
melbourne = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

melbourne.columns
print(melbourne.isnull().sum())     
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



mel_target = melbourne.Price

mel_predictors = melbourne.drop(['Price'],axis=1)



#Keep only numeric data

mel_numeric_predictors = mel_predictors.select_dtypes(exclude=['object'])
X_train, X_test, Y_train, Y_test = train_test_split(mel_target,mel_predictors,train_size=0.7,test_size=0.3,random_state=0)

def score_dataset(X_train, X_test, Y_train, Y_test):

    model = RandomForestRegressor()

    model.fit(X_train,Y_train)

    prediction = model.predict(X_test)

    return mean_absolute_error(Y_test,prediction)

cols_with_missing = [col for col in X_train.columns

                                 if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing)

reduced_X_test  = X_test.drop(cols_with_missing)

print("Mean Absolute Error from dropping columns with Missing Values:")

print(score_dataset(reduced_X_train, reduced_X_test, Y_train, Y_test))

    
from sklearn.preprocessing import Imputer



my_imputer = Imputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_test = my_imputer.transform(X_test)

imputed_X_train = np.reshape(2,1)

imputed_X_test = np.reshape(2,1)

print("Mean Absolute Error from Imputation:")

print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
import pandas as pd

train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



train_data.dropna(axis=0,subset=['SalePrice'],inplace=True)

target = train_data.SalePrice

cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

candidate_train_predictors = train_data.drop(['Id','SalePrice']+cols_with_missing,axis=1)

candidate_test_predictors = test_data.drop(['Id']+cols_with_missing,axis=1)



low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 

                        candidate_train_predictors[cname].nunique() < 10 and

                        candidate_train_predictors[cname].dtype == 'object']

numeric_cols = [cname for cname in candidate_test_predictors.columns if

                candidate_test_predictors[cname].dtype in ['int64','float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]

test_predictors = candidate_test_predictors[my_cols]

train_predictors.dtypes
one_hot_encoded_training = pd.get_dummies(train_predictors)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor



def get_mae(X,Y):

    return -1*cross_val_score(RandomForestRegressor(50),X,Y,

                              scoring='neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals,target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training,target)

print(mae_without_categoricals)

print(mae_one_hot_encoded)
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

final_train, final_test = one_hot_encoded_training_predictors.align(

    one_hot_encoded_test_predictors,join='left',axis=1)

mae = get_mae(final_train,target)

print(mae)
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer



train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



train_data.dropna(axis=0,subset=['SalePrice'],inplace=True)

Y = train_data.SalePrice

X = train_data.drop(['SalePrice'],axis=1).select_dtypes(exclude=['object'])



X_train, X_test, Y_train, Y_test = train_test_split(X.as_matrix(),Y.as_matrix(),test_size=0.25)



my_imputer = Imputer()

X_train = my_imputer.fit_transform(X_train)

X_test = my_imputer.transform(X_test)
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estmators=1000,learning_rate=0.05)

my_model.fit(X_train,Y_train,verbose=False,early_stopping_rounds=5,eval_set=[(X_test,Y_test)])
#predictions



prediction = my_model.predict(X_test)

mae = mean_absolute_error(prediction,Y_test)

print(mae)
def get_some_data():

    cols_to_use = ['Distance','Landsize','BuildingArea']

    data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

    Y = data.Price

    X = data[cols_to_use]

    my_imputer = Imputer()

    imputed_X = my_imputer.fit_transform(X)

    return imputed_X, Y

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.ensemble import GradientBoostingRegressor

X,Y = get_some_data()

my_model = GradientBoostingRegressor()

my_model.fit(X,Y)

my_plots = plot_partial_dependence(my_model,features=[0,2],X=X,

                                   feature_names=['Distance','Landsize','BuildingArea'],

                                   grid_resolution=10)



import pandas as pd

from sklearn.model_selection import train_test_split



data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

cols_to_use = ['Rooms','Distance','Landsize','BuildingArea','YearBuilt']

X = data[cols_to_use]

Y = data.Price

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer



my_pipeline = make_pipeline(Imputer(),RandomForestRegressor())

my_pipeline.fit(X_train, Y_train)

predictions = my_pipeline.predict(X_test)
import pandas as pd

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

cols_to_use = ['Rooms','Distance','Landsize','BuildingArea','YearBuilt']

X = data[cols_to_use]

Y = data.Price

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

my_pipeline=make_pipeline(Imputer(),RandomForestRegressor())

from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline,X,Y,scoring="neg_mean_absolute_error")

print(scores)
print('Mean Absolute Error %2f' %(-1*scores.mean()))