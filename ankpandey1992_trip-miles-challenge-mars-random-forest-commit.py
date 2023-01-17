import pandas as pd

from sklearn.feature_extraction import FeatureHasher

import numpy as np

import statsmodels.api as sm

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn import metrics

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.ensemble import AdaBoostRegressor

from pyearth import Earth
aggregated_data = pd.read_csv("../input/raw_aggregated_filteredCompany.csv")
aggregated_data.isnull().sum()
aggregated_data['tolls'].fillna((aggregated_data['tolls'].median()), inplace=True)

aggregated_data['fare'].fillna((aggregated_data['fare'].median()), inplace=True)

aggregated_data['tips'].fillna((aggregated_data['tips'].median()), inplace=True)

aggregated_data['extras'].fillna((aggregated_data['extras'].median()), inplace=True)

aggregated_data['trip_total'].fillna((aggregated_data['trip_total'].median()), inplace=True)
dummy = pd.get_dummies(aggregated_data.month_of_year, prefix='Month_flag')

aggregated_data = pd.concat([aggregated_data,dummy], axis = 1)



dummy = pd.get_dummies(aggregated_data.day_of_week, prefix='Day_week_flag')

aggregated_data = pd.concat([aggregated_data,dummy], axis = 1)



dummy = pd.get_dummies(aggregated_data.payment_type, prefix='Card_Type_flag')

aggregated_data = pd.concat([aggregated_data,dummy], axis = 1)
aggregated_data = aggregated_data[aggregated_data.fare <= 500]
#First chech the index of the features and label

list(zip( range(0,len(aggregated_data.columns)),aggregated_data.columns))
index=['peak_hours_flag','day_hours_flag','night_hours_flag','pickup_census_tract','dropoff_census_tract','pickup_community_area','dropoff_community_area',

      'tolls','fare','tips','extras','trip_total','trip_miles','Month_flag_1','Month_flag_2','Month_flag_3','Month_flag_4','Month_flag_5','Month_flag_6','Month_flag_7','Month_flag_8','Month_flag_9','Month_flag_10',

      'Month_flag_11','Month_flag_12','Day_week_flag_1','Day_week_flag_2','Day_week_flag_3','Day_week_flag_4','Day_week_flag_5','Day_week_flag_6','Day_week_flag_7','Card_Type_flag_Cash','Card_Type_flag_Credit Card',

      'Card_Type_flag_Dispute','Card_Type_flag_Mobile','Card_Type_flag_No Charge','Card_Type_flag_Pcard','Card_Type_flag_Prcard','Card_Type_flag_Prepaid','Card_Type_flag_Split','Card_Type_flag_Unknown',

      'Card_Type_flag_Way2ride']

X = aggregated_data[index].values

Y = aggregated_data.iloc[:,16].values
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=4, test_size=0.2)
model = Earth(max_terms=38)

model.fit(X_train,y_train) 
y_pred_mars = model.predict(X_test)
print('MAPE for the Multiple LR raw is : {}'.format(np.mean(np.abs((y_test-y_pred_mars)/y_test))*100))

print("\n")
#instantiate the object for the Random Forest Regressor with default params from raw data

regressor_rfraw = RandomForestRegressor(n_jobs=-1)







# #instantiate the object for the Random Forest Regressor with tuned hyper parameters 

# regressor_rf1 = RandomForestRegressor(n_estimators = 26,

#                                      max_depth = 22,

#                                      min_samples_split = 9,

#                                      n_jobs=-1)









#Train the object with default params for raw data

regressor_rfraw.fit(X_train,y_train)





print("\n")
#Predict the output with object of default params for Feature Selection Group

y_pred_rfraw = regressor_rfraw.predict(X_test)
#Evaluate the model with default params for raw data

print('MAPE for the RF regressor raw is : {}'.format(np.mean(np.abs((y_test-y_pred_rfraw)/y_test))*100))