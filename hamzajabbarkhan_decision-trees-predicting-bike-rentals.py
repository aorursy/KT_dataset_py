import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np 

%matplotlib inline 
bike_rentals = pd.read_csv('/kaggle/input/bike_rental_hour.csv')
bike_rentals.head()
bike_rentals.shape
plt.hist(bike_rentals['cnt'])

plt.show()
correlation = bike_rentals.corr()
correlation['cnt']
bike_rentals.info()
bike_rentals['hr'].unique()
def assign_label(hour): 

    if hour >= 0 and hour < 6: 

        return 4

    elif hour >= 6 and hour < 12: 

        return 1 

    elif hour >= 12 and hour < 18: 

        return 2 

    elif hour >= 18 and hour <= 24: 

        return 3
bike_data = bike_rentals.copy() 
bike_data['hr'] = bike_data['hr'].apply(assign_label)
bike_data['hr'].unique()
train_data = bike_data.sample(frac = 0.8)

#this is sampling WITHOUT replacement

train_data.head() 
train_data.shape
test_data = bike_data.iloc[~bike_data.index.isin(train_data.index)]
test_data.head()
test_data.shape
correlation['cnt']
cnt_correlation = correlation['cnt']

cnt_correlation
columns = cnt_correlation[(cnt_correlation > 0.3) & (cnt_correlation < 1)].index.values.tolist()
columns
columns.append('hum') 
columns
del columns[3:5]
columns
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
lr = LinearRegression() 

lr.fit(train_data[columns], train_data['cnt'])

predictions = lr.predict(test_data[columns])

mse = mean_squared_error(test_data['cnt'], predictions)

print(mse)
columns_2 = list(train_data.columns)
columns_2
columns_2.remove('dteday') 

columns_2.remove('cnt')

columns_2.remove('casual') 

columns_2.remove('registered')
columns_2
lr = LinearRegression() 

lr.fit(train_data[columns_2], train_data['cnt'])

predictions_2 = lr.predict(test_data[columns_2])

mse_2 = mean_squared_error(test_data['cnt'], predictions_2)

print(mse_2)
error_df = pd.DataFrame({'Model': ['LR - Restricted features','LR - Added Features'], 

                         'Error Value' : [mse,mse_2]}) 
error_df
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor() 

#we will use columns_2 as our feature set

dt.fit(train_data[columns_2], train_data['cnt'])

predictions_3 = dt.predict(test_data[columns_2])

error = mean_squared_error(test_data['cnt'], predictions_3)

print(error)
error_df = error_df.append({'Error Value' : error, 'Model' : 'Decision Tree - Same columns as LR Added Features'}, ignore_index=True)
error_df
dt2 = DecisionTreeRegressor(min_samples_leaf = 5)

dt2.fit(train_data[columns_2], train_data['cnt'])

predictions_4 = dt2.predict(test_data[columns_2])

error_2 = mean_squared_error(test_data['cnt'], predictions_4)

print(error_2)

#we will use SF for same features

error_df = error_df.append({'Error Value' : error_2, 'Model' : 'Decision Tree - SF, Min Sample Leaf - 5'}, ignore_index=True)
error_df
from sklearn.ensemble import RandomForestRegressor
dt3 = RandomForestRegressor(min_samples_leaf = 5)

dt3.fit(train_data[columns_2], train_data['cnt'])

predictions_5 = dt3.predict(test_data[columns_2])

error_3 = mean_squared_error(test_data['cnt'], predictions_5)

print(error_3)
error_df = error_df.append({'Error Value' : error_3, 'Model' : 'Random Forest - SF Min Sample Leaf - 5'}, ignore_index = True )
error_df