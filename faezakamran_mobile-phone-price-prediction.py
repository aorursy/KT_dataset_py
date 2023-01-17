# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
phones = pd.read_csv("../input/gsmarena-phone-dataset/phone_dataset .csv",error_bad_lines=False)
phones.head()
phones.info()
#cols to be used
cols = ['brand','model', 'GPRS', 'EDGE', 'status', 'dimentions', 'SIM', 'display_type',
                    'display_resolution', 'display_size', 'OS', 'CPU', 'Chipset', 'GPU', 'memory_card',
                    'internal_memory', 'RAM', 'primary_camera', 'secondary_camera', 'WLAN', 'bluetooth',
                    'GPS', 'sensors', 'battery', 'colors', 'approx_price_EUR']
#Creating a copy of the phone dataset with selected features and target
data = phones[cols]
data.head(20)
#number of unique values in each column
for col in data.columns:
    print(col, ':', data[col].nunique(), 'labels')
from pandas import Series

#function to split '|' separated values into multiple rows
def split_column_data_to_multiple_rows(df, col):
    s = df[col].str.split('|').apply(Series, 1).stack()
    s.index = s.index.droplevel(-1)
    s.name = col
    
    new_df = add_new_col_to_df(df, col, s)
    
    return new_df
#function to replace new column with the old one of the same name
def add_new_col_to_df(df, col, s):
    del df[col]
    new_df = df.join(s)
    
    return new_df
#creating a separate shorter dataframe to keep things simpler
#keeping the columns I will be working on first, and will keep adding others on the way
data_copy = data.copy()
data_df_cols = ['brand','model', 'primary_camera', 'secondary_camera', 'WLAN', 'bluetooth', 'sensors',
               'colors', 'approx_price_EUR']
data_df = data_copy[data_df_cols]
data_df.head()
#Split and replace with new column values for Primary Camera Column
new_primary_camera_df = split_column_data_to_multiple_rows(data_df, 'primary_camera')
new_primary_camera_df.drop_duplicates(keep=False,inplace=True) 
new_primary_camera_df.head()
#Split and replace with new column values for Secondary Camera Column
new_secondary_camera_df = split_column_data_to_multiple_rows(new_primary_camera_df, 'secondary_camera')
new_secondary_camera_df.drop_duplicates(keep=False,inplace=True) 
new_secondary_camera_df.head()
#Split and replace with new column values for WLAN Column
new_WLAN_df = split_column_data_to_multiple_rows(new_secondary_camera_df, 'WLAN')
new_WLAN_df.drop_duplicates(keep=False,inplace=True) 
new_WLAN_df.head()
#Split and replace with new column values for Bluetooth Column
new_bluetooth_df = split_column_data_to_multiple_rows(new_WLAN_df, 'bluetooth')
new_bluetooth_df.drop_duplicates(keep=False,inplace=True) 
new_bluetooth_df.head()
#Split and replace with new column values for Sensors Column
new_sensors_df = split_column_data_to_multiple_rows(new_bluetooth_df, 'sensors')
new_sensors_df.drop_duplicates(keep=False,inplace=True) 
new_sensors_df.head()
#Split and replace with new column values for Colors Column
new_colors_df = split_column_data_to_multiple_rows(new_sensors_df, 'colors')
new_colors_df.drop_duplicates(keep=False,inplace=True) 
new_colors_df.head(10)
#Bringig down internal memory column to the new dataframe for processing
new_df = new_colors_df.join(data_copy['internal_memory'])
new_df.head(10)
#Split and replace with new column values for Internal Memory Column
new_internal_memory_df = split_column_data_to_multiple_rows(new_df, 'internal_memory')
new_internal_memory_df.drop_duplicates(keep=False,inplace=True) 
new_internal_memory_df.head(10)
#Now bringing down status from original dataframe and joining to the updated dataframe
status_df = new_internal_memory_df.join(data_copy['status'])
status_df.head()
#Splitting Status Column into multiple status related columns
split_data = status_df["status"].str.split(" ")
sdata = split_data.to_list()
names = ["release_status", "released", "release_year", 'release_day', 'release_month/quarter', 'release_hour', 'release_min']
new_split_df = pd.DataFrame(sdata, columns=names)
new_status_df = new_split_df.drop(['released', 'release_day', 'release_hour', 'release_min'], axis=1)
new_status_df.head(10)
#The new dataframe contains newly created status related columns replacing the status column
#Notice that there is no status column
new_data_df = status_df.join(new_status_df)
new_data_df = new_data_df.drop(['status'], axis=1)
new_data_df.drop_duplicates(keep=False,inplace=True) 
new_data_df.head()
#Adding battery column to new dataframe for further processing
battery_df = new_data_df.join(data_copy['battery'])
battery_df.head()
#spliting battery column into multiple columns replacing the battery column
#Notice that there is no battery column
split_battery_data = battery_df['battery'].str.split(" ")
battery_data = split_battery_data.to_list()
battery_col_names = ['removable/non-removable', 'battery_type', 'battery_current', 'battery_unit', 'colname_battery', 'col6',
         'col7', 'col8', 'col9']
battery_split_df = pd.DataFrame(battery_data, columns=battery_col_names)
battery_split_df = battery_split_df.drop([ 'colname_battery','col6', 'col7', 'col8', 'col9'], axis=1)
new_battery_split_df = battery_split_df.replace(to_replace ="battery", value ="NaN") 
new_battery_split_df.head()
#adding newly created battery related columns to the dataframe
new_battery_data_df = new_data_df.join(new_battery_split_df)
new_battery_data_df.drop_duplicates(keep=False,inplace=True) 
new_battery_data_df.head()
#adding remaining features to the dataframe
full_df = new_battery_data_df.join(data_copy[['display_resolution', 'display_size', 'GPRS', 'EDGE',
                                              'dimentions', 'SIM', 'OS', 'CPU', 'Chipset', 'GPU',
                                              'memory_card', 'RAM', 'GPS']])
full_df.head(20)
#Stripping off unwanted information from display columns
full_df['display_resolution'] = full_df['display_resolution'].str.split('(').str[0]
full_df['display_size'] = full_df['display_size'].str.split('(').str[0]
full_df['dimentions'] = full_df['dimentions'].str.split('(').str[0]
full_df.drop_duplicates(keep=False,inplace=True) 
full_df.head()
#OS data contains "|" separated values thus splitting into multiples rows. 
#We now have the final dataframe containing all desired features
final_X_df = split_column_data_to_multiple_rows(full_df, 'OS')
final_X_df.drop_duplicates(keep=False,inplace=True) 
final_X_df.head(20)
from sklearn.model_selection import train_test_split

# Dataset with imputation
Y = final_X_df['approx_price_EUR'].fillna(final_X_df['approx_price_EUR'].mean()).values # Target for the model
#Y = final_X_df['approx_price_EUR']
X = final_X_df.drop(['approx_price_EUR'], axis=1) # Features we use

# splitting into two sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=0)
import category_encoders as ce

target_enc = ce.CatBoostEncoder()
target_enc.fit(X_train, Y_train)

# Transform the features, rename columns with _cb suffix, and join to dataframe
train_CBE = target_enc.transform(X_train)
test_CBE = target_enc.transform(X_test)
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_CBE))
imputed_X_test = pd.DataFrame(my_imputer.transform(test_CBE))

# Imputation removed column names; put them back
imputed_X_train.columns = train_CBE.columns
imputed_X_test.columns = test_CBE.columns
#check for null values in train set
train_CBE.isnull().sum()
#check for null values in test set
test_CBE.isnull().sum()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

lr_model = LinearRegression()
lr_model.fit(imputed_X_train, Y_train)
predictions = lr_model.predict(imputed_X_test)

mae = mean_absolute_error(predictions, Y_test)
r2score = r2_score(Y_test, predictions)

print("Validation MAE for Linear Regression Model: {}".format(mae))
print("Validation Accuracy for Linear Regression Model: {}".format(r2score))

output = pd.DataFrame({'Actual': Y_test, 'Predicted': predictions})
output.head(20)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# To improve accuracy, create a new Random Forest model and train on the data
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(imputed_X_train, Y_train)
predictions = rf_model.predict(imputed_X_test)

mae = mean_absolute_error(predictions, Y_test)
r2score = r2_score(Y_test, predictions)

print("Validation MAE for Random Forest Model: {}".format(mae))
print("Validation Accuracy for Random Forest Model: {}".format(r2score))

output = pd.DataFrame({'Actual': Y_test, 'Predicted': predictions})
output.head(20)
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=1000)
# Add silent=True to avoid printing out updates with each cycle
xgb_model.fit(imputed_X_train, Y_train, verbose=False)
predictions = xgb_model.predict(imputed_X_test)

mae = mean_absolute_error(predictions, Y_test)
r2score = r2_score(Y_test, predictions)

print("Validation MAE for XGBoost Model: {}".format(mae))
print("Validation Accuracy for XGBoost Model: {}".format(r2score))

output = pd.DataFrame({'Actual': Y_test, 'Predicted': predictions})
output.head(20)