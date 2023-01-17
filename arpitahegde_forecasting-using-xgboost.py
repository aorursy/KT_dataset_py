import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

#read only part of data
data=pd.read_csv('../input/Historical Product Demand.csv',nrows=100000)
data.dropna(axis=0)
predictors=['Product_Code','Warehouse','Product_Category','Date']
X=data[predictors]
y=data.Order_Demand
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1,test_size=0.1,train_size=0.5)
one_hot_encoding_val_data=pd.get_dummies(val_X)
one_hot_encoding_train_data=pd.get_dummies(train_X)
del val_X
del train_X
final_train, final_val = one_hot_encoding_train_data.align(one_hot_encoding_val_data,join='left',axis=1)
#final val and final train are X 
del one_hot_encoding_val_data
del one_hot_encoding_train_data

my_imputer = Imputer()
final_train = my_imputer.fit_transform(final_train)
final_val = my_imputer.transform(final_val)
#define model 
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(final_train, train_y, verbose=False)
del final_train
del train_y
# make predictions
predictions = my_model.predict(final_val)
del final_val
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, val_y)))