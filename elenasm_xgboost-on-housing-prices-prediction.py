

import numpy as np 

import pandas as pd 

from xgboost import XGBRegressor

from sklearn.datasets import load_boston



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# load dataset

house_price = load_boston()

df_labels = pd.DataFrame(house_price.target)

df = pd.DataFrame(house_price.data)

print(df_labels.head())

print(df.head())
df_labels.columns = ['PRICE']

df.columns = house_price.feature_names

print(df_labels.head())
df_total = df.merge(df_labels, left_index = True, right_index = True)

df_total.head()
df = preprocessing.scale(df)

X_train, X_test, y_train, y_test = train_test_split(

    df, df_labels, test_size=0.3, random_state=10)
#XGBoost part here!



my_model = XGBRegressor()

my_model.fit(X_train, y_train, verbose=False)
#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

rmse 
#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

rmse 
#n_estimators usually varies between 100 and 1000 so let's try it:



my_model = XGBRegressor(n_estimators = 100)

my_model.fit(X_train, y_train, verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
#n_estimators 2nd option, 200:



my_model = XGBRegressor(n_estimators = 200)

my_model.fit(X_train, y_train, verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
#n_estimators 3rd option, 500:



my_model = XGBRegressor(n_estimators = 500)

my_model.fit(X_train, y_train, verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
#n_estimators 4th option, 1000:



my_model = XGBRegressor(n_estimators = 1000)

my_model.fit(X_train, y_train, verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
# starting with a minimum of 5 early stopping rounds:

my_model = XGBRegressor(n_estimators = 1000)

my_model.fit(X_train, y_train,early_stopping_rounds=5,eval_set=[(X_test, y_test)], verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
# continuing with 15 early stopping rounds:

my_model = XGBRegressor(n_estimators = 1000)

my_model.fit(X_train, y_train,early_stopping_rounds=15,eval_set=[(X_test, y_test)], verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
# continuing with 15 early stopping rounds:

my_model = XGBRegressor(n_estimators = 1000)

my_model.fit(X_train, y_train,early_stopping_rounds=50,eval_set=[(X_test, y_test)], verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
# continuing with ( too much) 100 early stopping rounds:

my_model = XGBRegressor(n_estimators = 1000)

my_model.fit(X_train, y_train,early_stopping_rounds=100,eval_set=[(X_test, y_test)], verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
# starting with a small learning rate:

my_model = XGBRegressor(n_estimators = 1000,learning_rate=0.05)

my_model.fit(X_train, y_train,early_stopping_rounds=100,eval_set=[(X_test, y_test)], verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
# increasing the learning rate:

my_model = XGBRegressor(n_estimators = 1000,learning_rate=0.1)

my_model.fit(X_train, y_train,early_stopping_rounds=100,eval_set=[(X_test, y_test)], verbose=False)



#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
my_model = XGBRegressor(n_estimators = 1000,learning_rate=0.15)

my_model.fit(X_train, y_train,early_stopping_rounds=100,eval_set=[(X_test, y_test)], verbose=False)

#early_stopping_rounds can very well be kept at a much lower value, as it doesn't make a big difference

#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = my_model.predict(X_train)

mse = mean_squared_error(y_train_predicted, y_train)

rmse = np.sqrt(mse)

print(rmse)



#on test set

y_test_predicted = my_model.predict(X_test)

mse = mean_squared_error(y_test_predicted, y_test)

rmse = np.sqrt(mse)

print(rmse)
df_labels.describe()