













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
#Now the CSV files and take create the data frame for the appropiate 

house_price_data = pd.read_csv("../input/train.csv")

house_price_predict_data = pd.read_csv("../input/test.csv")

#Checking the data what kind of data and what are the columns of the data

print("Description of the train data")

print(house_price_data.describe())

print("Initial column values of the train data")

print(house_price_data.head())
#Checking what are the columns and their data types

col_type = {}

for col in house_price_data.columns:    

   # col_type[col] = house_price_data[col].dtype     

    print("Columns:{0} and Type={1}".format(col,house_price_data[col].dtype))

        

    

    

#print(col_type)
#Choosing the column in which the data have to processed

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#Creating the X,y data

X = house_price_data[features]

y = house_price_data.SalePrice
#Creating the train test split for the prepare the data

from sklearn.model_selection import train_test_split



train_X,test_X,train_y,test_y = train_test_split(X,y,random_state=1)
#Create the Descision tree regression for the model

from sklearn.tree import DecisionTreeRegressor

dtr_model = DecisionTreeRegressor(random_state=1)

dtr_model.fit(train_X,train_y)
#Predicting the value and check the mean square error 

from sklearn.metrics import mean_absolute_error

prediction = dtr_model.predict(test_X)
error = mean_absolute_error(test_y,prediction)

print(error)
#Now doing the performence tuning for the above data set

max_leaf_nodes = [10,50,100,500,1000,5000]

def check_performence_of_dtr(train_X,train_y,test_x,test_y,max_leaf_nodes,random_state):

    regression_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state = random_state )

    regression_model.fit(train_X,train_y)

    prediction = regression_model.predict(test_x)

    error = mean_absolute_error(test_y,prediction)

    return error





#Checking the value for the performence of the descison tree

random = 0

lst_for_zero = {}

for i in max_leaf_nodes:

    lst_for_zero[i]=(check_performence_of_dtr(train_X,train_y,test_X,test_y,i,random))

    

print("Error values for the random state = 0")

print(lst_for_zero)
random = 1

lst_for_one = {}

for i in max_leaf_nodes:

    lst_for_one[i]=(check_performence_of_dtr(train_X,train_y,test_X,test_y,i,1))

print("Error values for the random state = 1")

print(lst_for_one)
#Now checking the Another regression method 

from sklearn.ensemble import RandomForestRegressor

def checking_for_random_forest(train_X,train_y,test_x,test_y,random_state):

    regression_model = RandomForestRegressor(random_state = random_state)

    regression_model.fit(train_X,train_y)

    prediction = regression_model.predict(test_x)

    error = mean_absolute_error(test_y,prediction)

    return error

    
#checking the random forest error for different random state

state = [0,1]

lst_for_random_forest = {}

for i in state:

    lst_for_random_forest[i]=checking_for_random_forest(train_X,train_y,test_X,test_y,i)

    

print(lst_for_random_forest)

    
from xgboost import XGBRegressor

def check_XGBRegressor_tuning(train_X,train_y,test_X,test_y):

    model = XGBRegressor()

    model.fit(train_X,train_y)

    predict = model.predict(test_X)

    error = mean_absolute_error(predict,test_y)

    return error
#checking the error in the XGBRegessor

error = check_XGBRegressor_tuning(train_X,train_y,test_X,test_y)

print(error)
#Creating the Pipe line and imputer check the results for the both the random forest descision tree and xgboost

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

my_pipeline  = make_pipeline(Imputer(),RandomForestRegressor(random_state = 1))
my_pipeline.fit(train_X,train_y)

pred =my_pipeline.predict(test_X)
error  = mean_absolute_error(pred,test_y)

print(error)
#Checking different model and checking there mea

def check_different_models(train_X,train_y,test_X,test_y,model):

    my_pipeline = make_pipeline(Imputer(),model)

    my_pipeline.fit(train_X,train_y)

    pred = my_pipeline.predict(test_X)

    error = mean_absolute_error(pred,test_y)

    return error

    
list_model = [DecisionTreeRegressor(max_leaf_nodes=100,random_state = 0 ),RandomForestRegressor(random_state = 1), XGBRegressor()]

k = 1

error_per_model = {}

for i in list_model:

    error_per_model[k]= check_different_models(train_X,train_y,test_X,test_y,i)

    k = k+1

    

print(error_per_model)