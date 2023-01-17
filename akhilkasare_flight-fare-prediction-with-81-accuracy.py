# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_excel('/kaggle/input/flight-fare-prediction-mh/Data_Train.xlsx')

test_data = pd.read_excel('/kaggle/input/flight-fare-prediction-mh/Test_set.xlsx')
train_data.head()
test_data.head()
train_data.info()
test_data.info()
train_data.describe(include='O')
test_data.describe(include='O')
print("The shape of the train dataset is :", train_data.shape)

print("The shape of the test dataset is :", test_data.shape)
train_data.isnull().any()
test_data.isnull().any()
train_data.isnull().sum()
train_data.dropna(inplace=True)
train_data.shape
train_data.head()
train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'], format = '%d/%m/%Y').dt.day

train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'], format = '%d/%m/%Y').dt.month
train_data.head()
train_data.drop('Date_of_Journey', axis=1, inplace=True)
# Similarly we have to extraxt hours and minutes from the Dep_Time feature



# Extracting hour

train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour



# Extracting minutes

train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute



# Dropping the old Dep_Time column



train_data.drop(['Dep_Time'], axis=1, inplace=True)
train_data.head()
# Prerforming same actions for 'Arrival_Time' column



# Extracting hours

train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour



# Extracting minutes

train_data['Arrival_mins'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute



# Dropping the old column

train_data.drop(['Arrival_Time'], axis=1, inplace = True)
train_data.head()
# Time taken by plane to reach destination is called Duration

# It is the differnce betwwen Departure Time and Arrival time





# Assigning and converting Duration column into list

duration = list(train_data["Duration"])



for i in range(len(duration)):

    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins

        if "h" in duration[i]:

            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute

        else:

            duration[i] = "0h " + duration[i]           # Adds 0 hour



duration_hours = []

duration_mins = []

for i in range(len(duration)):

    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration

    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
# Adding duration_hours and duration_mins list to train_data dataframe



train_data["Duration_hours"] = duration_hours

train_data["Duration_mins"] = duration_mins
train_data.drop(['Duration'], axis=1, inplace=True)
train_data.head()
train_data.Airline.value_counts()
plt.figure(figsize=(10,10))

sns.barplot(x = 'Airline', y = 'Price', data=train_data)

plt.legend()

plt.xticks( rotation=90)
sns.distplot(train_data['Price'])
train_data.head()
train_data.Total_Stops.value_counts()
train_data.groupby('Airline')['Price'].mean()
plt.figure(figsize=(10,10))

sns.boxplot(x = 'Airline', y = 'Price', data=train_data)

plt.xticks( rotation=90)
train_data.groupby('Destination')['Price'].mean()
plt.figure(figsize=(10,10))

sns.boxplot(x = 'Destination', y = 'Price', data=train_data)

plt.xticks( rotation=45)
# Hadeling the categorical features



cat_features = [feature for feature in train_data.columns if train_data[feature].dtypes == 'O']

list(cat_features)
train_data.head()
train_data.Total_Stops.value_counts()
train_data.replace({'non-stop' : 0, '1 stop' : 1, '2 stops' : 2, '3 stops' : 3, '4 stops' : 4}, inplace = True)

train_data.head()
train_data.drop(['Route', 'Additional_Info'], axis = 1, inplace = True)
train_data.head()
train_data.Airline.value_counts()
train_data.head()
Airline = train_data[['Airline']]



Airline = pd.get_dummies(Airline, drop_first = True)
Source = train_data[['Source']]



Source = pd.get_dummies(Source, drop_first = True)
Destination = train_data[['Destination']]



Destination = pd.get_dummies(Destination, drop_first = True)
train_data.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace = True)

train_data.head()
df_train =  pd.concat([train_data, Airline, Source, Destination], axis = 1)
df_train.head()
df_train.isnull().any()
test_data.head()
# Extracting day and month from 'Date_of_Journey' column.



# Extracting day

test_data['Journey_day'] = pd.to_datetime(test_data['Date_of_Journey'], format="%d/%m/%Y").dt.day



# Extracting month

test_data['Journey_month'] = pd.to_datetime(test_data['Date_of_Journey'], format="%d/%m/%Y").dt.month



# Dropping the old column

test_data.drop('Date_of_Journey',axis = 1, inplace = True)

test_data.head()
# Extracting hours and mins from 'Dep_Time' column



# Extracting hours

test_data['Dep_hour'] = pd.to_datetime(test_data['Dep_Time']).dt.hour



# Extracting minutes

test_data['Dep_mins'] = pd.to_datetime(test_data['Dep_Time']).dt.minute



# Dropping the old column

test_data.drop('Dep_Time', axis=1, inplace=True)

test_data.head()
# Prerforming same actions for 'Arrival_Time' column



# Extracting hours

test_data['Arrival_hour'] = pd.to_datetime(test_data['Arrival_Time']).dt.hour



# Extracting minutes

test_data['Arrival_mins'] = pd.to_datetime(test_data['Arrival_Time']).dt.minute



# Dropping the old column

test_data.drop(['Arrival_Time'], axis=1, inplace = True)
# Similarly extracting hours and minutes form the 'Arrival_Time' column





# It is the differnce betwwen Departure Time and Arrival time





# Assigning and converting Duration column into list

duration = list(test_data["Duration"])



for i in range(len(duration)):

    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins

        if "h" in duration[i]:

            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute

        else:

            duration[i] = "0h " + duration[i]           # Adds 0 hour



duration_hours = []

duration_mins = []

for i in range(len(duration)):

    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration

    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
# Adding duration_hours and duration_mins list to train_data dataframe



test_data["Duration_hours"] = duration_hours

test_data["Duration_mins"] = duration_mins
test_data.head()
test_data.Total_Stops.value_counts()
test_data.replace({'non-stop' : 0, '1 stop' : 1, '2 stops' : 2, '3 stops' : 3, '4 stops' : 4}, inplace=True)

test_data.head()
test_data.Total_Stops.value_counts()
Airline = test_data[['Airline']]



Airline = pd.get_dummies(Airline, drop_first=True)
Source = test_data[['Source']]



Source = pd.get_dummies(Airline, drop_first=True)
Destination = test_data[['Destination']]



Destination = pd.get_dummies(Destination, drop_first=True)
test_data.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace = True)
df_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

df_test.head()
df_test.drop(['Route', 'Additional_Info', 'Duration'], axis = 1, inplace=True)
df_test.head()
# Plotting a correlation matrix



plt.figure(figsize=(30, 30))

sns.heatmap(df_train.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
highly_corr_features = df_train[df_train.corr() > 0.7]



list(highly_corr_features)
df_train.columns
# Splitting the data into Independent and dependent variables



x = df_train[['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',

       'Dep_min', 'Arrival_hour', 'Arrival_mins', 'Duration_hours',

       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',

       'Airline_Jet Airways', 'Airline_Jet Airways Business',

       'Airline_Multiple carriers',

       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',

       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',

       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',

       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',

       'Destination_Kolkata', 'Destination_New Delhi']]



y = df_train['Price']
# Important feature using ExtraTreesRegressor



from sklearn.ensemble import ExtraTreesRegressor

selection = ExtraTreesRegressor()

selection.fit(x, y)
print(selection.feature_importances_)
#plot graph of feature importances for better visualization



plt.figure(figsize = (12,8))

feat_importances = pd.Series(selection.feature_importances_, index=x.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
from sklearn.tree import DecisionTreeRegressor



tree = DecisionTreeRegressor(random_state=42)

tree.fit(x_train, y_train)



dtree_pred = tree.predict(x_test)
tree.score(x_train, y_train)
tree.score(x_test, y_test)
sns.distplot(y_test-dtree_pred)
plt.scatter(y_test, dtree_pred, alpha = 0.5)

plt.xlabel("y_test")

plt.ylabel("dtree prediction")

plt.show()
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, dtree_pred))

print('MSE:', metrics.mean_squared_error(y_test, dtree_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtree_pred)))
metrics.r2_score(y_test, dtree_pred)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
rf.score(x_train, y_train)
rf.score(x_test, y_test)
sns.distplot(y_test-rf_pred)
plt.scatter(y_test, rf_pred, alpha = 0.5)

plt.xlabel("y_test")

plt.ylabel("dtree prediction")

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, rf_pred))

print('MSE:', metrics.mean_squared_error(y_test, rf_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))
metrics.r2_score(y_test, rf_pred)
from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
# Create the random grid



random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}

# Random search of parameters, using 5 fold cross validation, 

# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(x_train, y_train)
rf_random.best_params_
rf_prediction = rf_random.predict(x_test)
plt.figure(figsize = (8,8))

sns.distplot(y_test-rf_prediction)

plt.show()
plt.figure(figsize = (8,8))

plt.scatter(y_test, rf_prediction, alpha = 0.5)

plt.xlabel("y_test")

plt.ylabel("y_pred")

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, rf_prediction))

print('MSE:', metrics.mean_squared_error(y_test, rf_prediction))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_prediction)))
metrics.r2_score(y_test, rf_prediction)
import pickle

# open a file, where you ant to store the data

file = open('flight_rf.pkl', 'wb')



# dump information to that file

pickle.dump(rf, file)