# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('../input/housing.csv')
data.info()
print ("Description : \n\n", data.describe())
data.head()
data.hist(figsize=(20,15), color = 'green')
plt.show()
print('Let\'s check for null values\n')
print(data.isnull().sum())     
# Droping NaN value
data = data.dropna(axis=0)
print("\nNew Shape after dropping NULL value : ", data.shape)
print('Let\'s check for null values\n')
print(data.isnull().sum())  
# Dropping ['median_house_value', ocean_proximity]
x_data = data.drop(data.columns[[8, 9]], axis = 1)

y_data = data['median_house_value']
x_data.head()
y_data.head()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, random_state=101)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = pd.DataFrame(data = scaler.transform(x_train), columns = x_train.columns, index= x_train.index)
x_train.head()
x_test = pd.DataFrame(data = scaler.transform(x_test), columns = x_test.columns, index= x_test.index)
x_test.head()
data.columns
longitude = tf.feature_column.numeric_column('longitude')
latitutde = tf.feature_column.numeric_column('latitude')
age = tf.feature_column.numeric_column('housing_median_age')
rooms = tf.feature_column.numeric_column('total_rooms')
bedroom = tf.feature_column.numeric_column('total_bedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('median_income')
# Aggregating the feature columns
feat_cols = [longitude, latitutde, age, rooms, bedroom, pop, households, income]
feat_cols
input_func = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train, batch_size = 20, num_epochs = 2000, shuffle = True)
model = tf.estimator.DNNRegressor(hidden_units = [8, 8, 8, 8, 8], feature_columns = feat_cols)
model.train(input_fn = input_func, steps = 50000)
predict_input_func = tf.estimator.inputs.pandas_input_fn(x = x_test, batch_size = 20, num_epochs = 1, shuffle = False)
pred_gen = model.predict(predict_input_func)    
predictions = list(pred_gen) 
predictions
final_y_preds = []

for pred in predictions:
    final_y_preds.append(pred['predictions'])
final_y_preds
# Fianl RMSE Value using DNN Regressor
mean_squared_error(y_test, final_y_preds) ** 0.5
# Training Model
rf_regressor = RandomForestRegressor(n_estimators=500, random_state = 0)
rf_regressor.fit(x_train, y_train)
# Predicting the values
y_pred = rf_regressor.predict(x_test)
p = mean_squared_error(y_test, y_pred)
print(p ** 0.5)    