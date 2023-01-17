import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





sns.set()
pd.set_option('display.max_columns', None)

train = pd.read_excel(r"Train.xlsx",parse_dates=['Date_of_Journey','Arrival_Time'])

train.head()
train.info()
train.isna().sum()
train[train.Route.isna()]
train[(train.Source=='Delhi') & (train.Destination=='Cochin') &(train.Price==7480) & (train.Dep_Time=='09:45')]
train.Route.fillna('DEL → MAA → COK',inplace=True)

train.Total_Stops.fillna('1 stop',inplace=True)
train[train.Total_Stops.isna()]
train.Date_of_Journey.unique()
(train["Date_of_Journey"]).dt.day.unique()
train.head()
## Convert month & date from date of journey

train['Journey_day']=train['Date_of_Journey'].dt.day

train['Journey_month']=train['Date_of_Journey'].dt.month



# Departure time is when a plane leaves the gate. 

# Similar to Date_of_Journey we can extract values from Dep_Time



# Extracting Hours

train["Dep_hour"] = pd.to_datetime(train["Dep_Time"]).dt.hour



# Extracting Minutes

train["Dep_min"] = pd.to_datetime(train["Dep_Time"]).dt.minute



# Arrival time is when the plane pulls up to the gate.

# Similar to Date_of_Journey we can extract values from Arrival_Time



# Extracting Hours

train["Arrival_hour"] = train.Arrival_Time.dt.hour



# Extracting Minutes

train["Arrival_min"] = train.Arrival_Time.dt.minute
train.head()
# Since we have converted Date_of_Journey column into integers, Now we can drop as it is of no use.



train.drop(["Date_of_Journey"], axis = 1, inplace = True)





# Now we can drop Dep_Time as it is of no use

train.drop(["Dep_Time"], axis = 1, inplace = True)







# Now we can drop Arrival_Time as it is of no use

train.drop(["Arrival_Time"], axis = 1, inplace = True)
train.head()
# Time taken by plane to reach destination is called Duration

# It is the differnce betwwen Departure Time and Arrival time





# Assigning and converting Duration column into list

duration = list(train["Duration"])



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



train["Duration_hours"] = duration_hours

train["Duration_mins"] = duration_mins
train.drop(["Duration"], axis = 1, inplace = True)
train.head()
train["Airline"].value_counts()
# From graph we can see that Jet Airways Business have the highest Price.

# Apart from the first Airline almost all are having similar median



# Airline vs Price

sns.catplot(y = "Price", x = "Airline", data = train.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)

plt.show()
# As Airline is Nominal Categorical data we will perform OneHotEncoding



Airline = train[["Airline"]]



Airline = pd.get_dummies(Airline, drop_first= True)



Airline.head()
train["Source"].value_counts()
# Source vs Price



sns.catplot(y = "Price", x = "Source", data = train.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)

plt.show()
# As Source is Nominal Categorical data we will perform OneHotEncoding



Source = train[["Source"]]



Source = pd.get_dummies(Source, drop_first= True)



Source.head()
train["Destination"].value_counts()
# As Destination is Nominal Categorical data we will perform OneHotEncoding



Destination = train[["Destination"]]



Destination = pd.get_dummies(Destination, drop_first = True)



Destination.head()
train["Route"]
# Additional_Info contains almost 80% no_info

# Route and Total_Stops are related to each other



train.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
train["Total_Stops"].value_counts()
# As this is case of Ordinal Categorical type we perform LabelEncoder

# Here Values are assigned with corresponding keys



train.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
train.head()
# Concatenate dataframe --> train_data + Airline + Source + Destination

train = pd.concat([train, Airline, Source, Destination], axis = 1)
train.head()
train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
train.head()
train.shape
test = pd.read_excel("Test.xlsx")
test.head()
# Preprocessing



print("Test data Info")

print("-"*75)

print(test.info())



print()

print()



print("Null values :")

print("-"*75)

test.dropna(inplace = True)

print(test.isnull().sum())



# EDA



# Date_of_Journey

test["Journey_day"] = pd.to_datetime(test.Date_of_Journey, format="%d/%m/%Y").dt.day

test["Journey_month"] = pd.to_datetime(test["Date_of_Journey"], format = "%d/%m/%Y").dt.month

test.drop(["Date_of_Journey"], axis = 1, inplace = True)



# Dep_Time

test["Dep_hour"] = pd.to_datetime(test["Dep_Time"]).dt.hour

test["Dep_min"] = pd.to_datetime(test["Dep_Time"]).dt.minute

test.drop(["Dep_Time"], axis = 1, inplace = True)



# Arrival_Time

test["Arrival_hour"] = pd.to_datetime(test.Arrival_Time).dt.hour

test["Arrival_min"] = pd.to_datetime(test.Arrival_Time).dt.minute

test.drop(["Arrival_Time"], axis = 1, inplace = True)



# Duration

duration = list(test["Duration"])



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



# Adding Duration column to test set

test["Duration_hours"] = duration_hours

test["Duration_mins"] = duration_mins

test.drop(["Duration"], axis = 1, inplace = True)





# Categorical data



print("Airline")

print("-"*75)

print(test["Airline"].value_counts())

Airline = pd.get_dummies(test["Airline"], drop_first= True)



print()



print("Source")

print("-"*75)

print(test["Source"].value_counts())

Source = pd.get_dummies(test["Source"], drop_first= True)



print()



print("Destination")

print("-"*75)

print(test["Destination"].value_counts())

Destination = pd.get_dummies(test["Destination"], drop_first = True)



# Additional_Info contains almost 80% no_info

# Route and Total_Stops are related to each other

test.drop(["Route", "Additional_Info"], axis = 1, inplace = True)



# Replacing Total_Stops

test.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)



# Concatenate dataframe --> test + Airline + Source + Destination

test = pd.concat([test, Airline, Source, Destination], axis = 1)



test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)



print()

print()



print("Shape of test data : ", data_test.shape)



test.head()
test.shape
X= train.drop('Price',axis=1)

y=train.Price
X.head()
y.head()
# Finds correlation between Independent and dependent attributes



plt.figure(figsize = (18,18))

sns.heatmap(train.corr(), annot = True, cmap = "RdYlGn")



plt.show()
# Important feature using ExtraTreesRegressor



from sklearn.ensemble import ExtraTreesRegressor

selection = ExtraTreesRegressor()

selection.fit(X, y)
print(selection.feature_importances_)
#plot graph of feature importances for better visualization



plt.figure(figsize = (12,8))

feat_importances = pd.Series(selection.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.ensemble import RandomForestRegressor

reg_rf = RandomForestRegressor()

reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
reg_rf.score(X_train, y_train)
reg_rf.score(X_test, y_test)
sns.distplot(y_test-y_pred)

plt.show()


plt.scatter(y_test, y_pred, alpha = 0.5)

plt.xlabel("y_test")

plt.ylabel("y_pred")

plt.show()
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# RMSE/(max(DV)-min(DV))



2090.5509/(max(y)-min(y))
metrics.r2_score(y_test, y_pred)
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

rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
prediction = rf_random.predict(X_test)
plt.figure(figsize = (8,8))

sns.distplot(y_test-prediction)

plt.show()
plt.figure(figsize = (8,8))

plt.scatter(y_test, prediction, alpha = 0.5)

plt.xlabel("y_test")

plt.ylabel("y_pred")

plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, prediction))

print('MSE:', metrics.mean_squared_error(y_test, prediction))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
import pickle

# open a file, where you ant to store the data

file = open('flight_rf.pkl', 'wb')



# dump information to that file

pickle.dump(rf_random, file)
model = open('flight_rf.pkl','rb')

forest = pickle.load(model)
y_prediction = forest.predict(X_test)
metrics.r2_score(y_test, y_prediction)