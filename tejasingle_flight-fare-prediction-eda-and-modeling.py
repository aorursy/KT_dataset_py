# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





# Importing necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor
train_data= pd.read_excel('../input/flight-fare-prediction-data/Data_Train.xlsx')
train_data.head()
train_data.describe()

# Price is the only integer column

# all other column are object type
train_data.dropna(inplace = True)



# dropping the missing data from the original data

# checking for null values in the data

train_data.isnull().sum()
# Looking at Price distribution

sns.distplot(train_data['Price'])

plt.show()
# Price feature after log transformation.

sns.distplot(np.log(train_data['Price']))

plt.show()
train_data['Date_of_Journey'] = pd.to_datetime(train_data['Date_of_Journey'],format= "%d/%m/%Y")
# extracting information from 'date_of_journey' column and storing in new columns 'Journey_month' and 'journey_day'

train_data['Journey_month']= pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.month_name()





train_data['Journey_day']= pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.day_name()
# We have converted Date_of_Journey column into integer type by creating two new columns , we can drop 'Date_of_Journey'.



train_data.drop(["Date_of_Journey"],axis=1,inplace=True)
plt.figure(figsize=(10,5))

sns.countplot(x = 'Journey_month', data = train_data)

plt.show()

plt.figure(figsize=(10,5))

sns.countplot(x = 'Journey_day', data = train_data)

plt.show()
train_data.groupby(['Journey_month']).mean()

#Here,We can see that the prices are the highest in March while the price for a flight in april is the least


train_data.groupby(['Journey_day']).mean()

#we can say that the highest prices are on Friday while lowest being on Monday.
# Extracting Hours from 'Dep_Time' column by creating a new column 'Dep_hour'

train_data["Dep_hour"]= pd.to_datetime(train_data['Dep_Time']).dt.hour



#  creating a new column 'Dep_min'

train_data["Dep_min"]= pd.to_datetime(train_data['Dep_Time']).dt.minute



# Now we can drop Dep_time column 

train_data.drop(["Dep_Time"],axis=1,inplace=True)
# creating a new column 'Arr_hour'

train_data["Arr_hour"]= pd.to_datetime(train_data['Arrival_Time']).dt.hour



# creating a new column 'Arr_min'

train_data["Arr_min"]= pd.to_datetime(train_data['Arrival_Time']).dt.minute



# Now we can drop Arrival_time columns also as we extracted useful information from it

train_data.drop(["Arrival_Time"],axis=1,inplace=True)
train_data['Time_Zone'] = pd.cut(train_data.Dep_hour, [1,4,11,16,23], labels=["Late_Night","Morning","Afternoon","Evening"])



Time_Zone = train_data[["Time_Zone"]]



Time_Zone = pd.get_dummies(Time_Zone, drop_first= True)

Time_Zone.head()
train_data.head()
def hr_min(hr, o):

    if ('h' in o) and ('m' in o):

        h, m = o.split()

        h = int(h.strip('h'))

        m = int(m.strip('m'))

    elif 'h' in o:

        h = int(o.strip('h'))

        m = 0 

    else:

        m = int(o.strip('m'))

        h = 0 

        

    return h if hr else m
from functools import partial 

f = partial(hr_min, True)

train_data['duration_hr'] = train_data.Duration.map(f)

f = partial(hr_min, False)

train_data['duration_min'] = train_data.Duration.map(f)







train_data.drop(["Duration"],axis=1,inplace=True)
train_data.head()
train_data['Airline'].value_counts()

# AIRLINE vs PRICE

sns.catplot(y='Price',x='Airline',data= train_data.sort_values('Price',ascending=False),kind="bar",height=6, aspect=3)

plt.show()


Airline = train_data[['Airline']]



Airline = pd.get_dummies(Airline, drop_first= True)



Airline.head() 
# Source vs PRICE

sns.catplot(y='Price',x='Source',data= train_data.sort_values('Price',ascending=False),kind="bar",height=6, aspect=3)

plt.show()


Source = train_data[["Source"]]



Source = pd.get_dummies(Source, drop_first= True)



Source.head()
# Destination vs PRICE

sns.catplot(y='Price',x='Destination',data= train_data.sort_values('Price',ascending=False),kind="bar",height=6, aspect=3)

plt.show()
Destination = train_data[['Destination']]



Destination = pd.get_dummies(Destination, drop_first = True)



Destination.head()
train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

train_data["Total_Stops"].value_counts()

train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops":3, "4 stops": 4}, inplace = True)

train_data.head()
Journey_day = pd.get_dummies(train_data['Journey_day'], drop_first = True)

Journey_month = pd.get_dummies(train_data['Journey_month'], drop_first = True)

#Removing the following columns  as the information has already been extracted from them.



train_data.drop(["Airline", "Source", "Destination","Journey_month","Journey_day","Time_Zone"], inplace = True, axis = 1)
# Concatenate dataframe



Train= pd.concat([train_data,Airline,Source,Destination,Journey_month,Journey_day,Time_Zone],axis=1)

Train.head()
test_data= pd.read_excel('../input/flight-fare-prediction-test-data/Test_set.xlsx')
test_data.head()
test_data.shape
test_data.dtypes
test_data["Date_of_Journey"] = pd.to_datetime(test_data["Date_of_Journey"], format= "%d/%m/%Y")



#deriving day of journey and month of journey from this

test_data["Day_of_journey"] = test_data["Date_of_Journey"].dt.day_name()

test_data["Month_of_journey"] = test_data["Date_of_Journey"].dt.month_name()



#Removing date of journey from the test data

test_data.drop(['Date_of_Journey'], axis = 1, inplace = True)
test_data.head()
#for arrival time



# creating new variables according to the arrival hour and minute

test_data['arrival_hour'] = pd.to_datetime(test_data['Arrival_Time']).dt.hour

test_data['arrival_min'] = pd.to_datetime(test_data['Arrival_Time']).dt.minute



#dropping the Arrival time column as all the features have already been extracted from it

test_data.drop("Arrival_Time",axis = 1, inplace = True)
#for departure time 



# extracting departure hour and minute

test_data['departure_hour'] = pd.to_datetime(test_data['Dep_Time']).dt.hour



test_data['departure_min'] = pd.to_datetime(test_data['Dep_Time']).dt.minute



# removing the departure time column as all the information has been extracted from it 



test_data.drop("Dep_Time", axis = 1, inplace = True)
test_data['Time_Zone'] = pd.cut(test_data.departure_hour, [0,4,11,16,23], labels=["Late_Night","Morning","Afternoon","Evening"])



Time_Zone = test_data[["Time_Zone"]]



Time_Zone = pd.get_dummies(Time_Zone, drop_first= True)

Time_Zone.head()
def hr_min(hr, o):

    if ('h' in o) and ('m' in o):

        h, m = o.split()

        h = int(h.strip('h'))

        m = int(m.strip('m'))

    elif 'h' in o:

        h = int(o.strip('h'))

        m = 0 

    else:

        m = int(o.strip('m'))

        h = 0 

        

    return h if hr else m
from functools import partial 

f = partial(hr_min, True)

test_data['duration_hr'] = test_data.Duration.map(f)

f = partial(hr_min, False)

test_data['duration_min'] = test_data.Duration.map(f)







test_data.drop(["Duration"],axis=1,inplace=True)
#Dropping Route and additional information columns

test_data.drop(["Route", "Additional_Info","Time_Zone"], axis = 1, inplace = True)


test_data.replace({"non-stop": 0,"1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
#one hot encoding for Airline, Source and Destination





Airline = pd.get_dummies(test_data["Airline"], drop_first= True)





Source = pd.get_dummies(test_data["Source"], drop_first= True)





Destination = pd.get_dummies(test_data["Destination"], drop_first= True)



test_data.drop(["Airline","Source","Destination"], axis = 1, inplace = True)

test_data.head()
#one hot encoding for journey day and month for test data



journey_day = pd.get_dummies(test_data["Day_of_journey"], drop_first= True)

journey_month = pd.get_dummies(test_data["Month_of_journey"], drop_first= True)



#removing the extra columns

test_data.drop(["Day_of_journey","Month_of_journey"], axis = 1, inplace = True)
Test = pd.concat([test_data,Airline, Source, Destination, journey_day, journey_month,Time_Zone], axis = 1)

Test.shape

Test.head()
Train.columns
Train.head()
X = Train.loc[:,['Total_Stops', 'Dep_hour', 'Dep_min', 'Arr_hour', 'Arr_min',

       'duration_hr', 'duration_min', 'Airline_Air India', 'Airline_GoAir',

       'Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business',

       'Airline_Multiple carriers',

       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',

       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',

       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',

       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',

       'Destination_Kolkata', 'Destination_New Delhi', 'June', 'March', 'May',

       'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',

       'Time_Zone_Morning', 'Time_Zone_Afternoon', 'Time_Zone_Evening']]



X.head()

y = Train.iloc[:,1]

y.head()
#using heatmap to find the correlation of the variables



plt.figure(figsize = (20,10))

sns.heatmap(train_data.corr(), annot = True, cmap= "YlGnBu")



plt.show()
# Important features using ExtraTreeRegressor



from sklearn.ensemble import ExtraTreesRegressor

selection= ExtraTreesRegressor()

selection.fit(X,y)
plt.figure(figsize = (12,8)) 

feat_importances = pd.Series(selection.feature_importances_, index=X.columns) 

feat_importances.nlargest(20).plot(kind='barh') 

plt.show()
# Splitting the dataset into Traing and Testing.



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Using Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

reg_rf = RandomForestRegressor()

reg_rf.fit(X_train, y_train)
# prediction variable 'y_pred'

y_pred= reg_rf.predict(X_test)
# Accuracy to training sets

reg_rf.score(X_train,y_train)
# accuracy of Testing sets

reg_rf.score(X_test,y_test)
sns.distplot(y_test-y_pred)

plt.show()
plt.scatter(y_test,y_pred,alpha=0.5)

plt.xlabel("y_test")

plt.ylabel("y_pred")

plt.show()
from sklearn import metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# R square error

metrics.r2_score(y_test,y_pred)
#Randomized Search CV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 14)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 8)]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
# create random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}
# Random search of parameters, using 5 fold cross validation, 



from sklearn.model_selection import RandomizedSearchCV

rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
# looking at best parameters

rf_random.best_params_
predict = rf_random.predict(X_test)
plt.figure(figsize = (8,8))

sns.distplot(y_test-predict)

plt.show()
# plt.figure(figsize = (8,8))

plt.scatter(y_test, predict, alpha = 0.5)

plt.xlabel("y_test")

plt.ylabel("y_pred")

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, predict))

print('MSE:', metrics.mean_squared_error(y_test, predict))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))
metrics.r2_score(y_test,predict)