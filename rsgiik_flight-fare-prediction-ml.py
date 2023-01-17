import pandas as pd

import numpy as np

import seaborn as sns



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

py.init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.metrics import r2_score,mean_squared_error



from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor



training_set = pd.read_excel("../input/flight-fare-prediction-mh/Data_Train.xlsx")

test_set = pd.read_excel("../input/flight-fare-prediction-mh/Test_set.xlsx")

training_set.head() 
training_set.info() 
training_set.isnull().sum()
training_set = training_set.dropna()

#data_train = data_train.dropna()

#data_train.shape
test_set.info()
test_set.isnull().sum()
training_set.shape
training_set.isna().any()
training_set['Journey_Day'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.day

training_set['Journey_Month'] = pd.to_datetime(training_set.Date_of_Journey, format='%d/%m/%Y').dt.month

training_set.head()
# Test Set

test_set['Journey_Day'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.day

test_set['Journey_Month'] = pd.to_datetime(test_set.Date_of_Journey, format='%d/%m/%Y').dt.month

test_set.head()
training_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)

test_set.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)
training_set['Duration'].value_counts()
duration = list(training_set['Duration'])





for i in range(len(duration)) :

    if len(duration[i].split()) != 2: 

        if 'h' in duration[i] :

            duration[i] = duration[i].strip() + ' 0m'

        elif 'm' in duration[i] :

            duration[i] = '0h {}'.format(duration[i].strip())



dur_hours = []

dur_minutes = []  



for i in range(len(duration)) :

    dur_hours.append(int(duration[i].split()[0][:-1])) #for examole if duration is 49 mintutes 4 sec then it will reflect like 

    dur_minutes.append(int(duration[i].split()[1][:-1]))#0:49:4 and if 2 hours 10 seconds then it will reflect like 2:0:10

    

training_set['Duration_hours'] = dur_hours

training_set['Duration_minutes'] =dur_minutes



training_set.drop(labels = 'Duration', axis = 1, inplace = True) # dropping the original duration column from training set

duration = list(test_set['Duration'])





for i in range(len(duration)) :

    if len(duration[i].split()) != 2: 

        if 'h' in duration[i] :

            duration[i] = duration[i].strip() + ' 0m'

        elif 'm' in duration[i] :

            duration[i] = '0h {}'.format(duration[i].strip())



dur_hours = []

dur_minutes = []  



for i in range(len(duration)) :

    dur_hours.append(int(duration[i].split()[0][:-1])) #for examole if duration is 49 mintutes 4 sec then it will reflect like 

    dur_minutes.append(int(duration[i].split()[1][:-1]))#0:49:4 and if 2 hours 10 seconds then it will reflect like 2:0:10

    

test_set['Duration_hours'] = dur_hours

test_set['Duration_minutes'] =dur_minutes



test_set.drop(labels = 'Duration', axis = 1, inplace = True) # dropping the original duration column from training set

#Converting 'Dep_Time' to 'Depart_Time_hour' and 'Depart_time_Minutes'

training_set['Dep_hour'] = pd.to_datetime(training_set['Dep_Time']).dt.hour

training_set['Dep_min'] = pd.to_datetime(training_set['Dep_Time']).dt.minute

training_set.drop(labels='Dep_Time', axis = 1, inplace= True)
#test_set

#Converting 'Dep_Time' to 'Depart_Time_hour' and 'Depart_time_Minutes'

test_set['Dep_hour'] = pd.to_datetime(test_set['Dep_Time']).dt.hour

test_set['Dep_min'] = pd.to_datetime(test_set['Dep_Time']).dt.minute

test_set.drop(labels='Dep_Time', axis = 1, inplace= True)
#Converting 'Arrival_Time' to 'Arrival_Time_hour' and 'Arrival_time_Minutes'

training_set['Arrival_hour'] = pd.to_datetime(training_set['Arrival_Time']).dt.hour

training_set['Arrival_min'] = pd.to_datetime(training_set['Arrival_Time']).dt.minute

training_set.drop(labels='Arrival_Time', axis = 1, inplace= True)

training_set.head()
#test_set

#Converting 'Arrival_Time' to 'Arrival_Time_hour' and 'Arrival_time_Minutes'

test_set['Arrival_hour'] = pd.to_datetime(test_set['Arrival_Time']).dt.hour

test_set['Arrival_min'] = pd.to_datetime(test_set['Arrival_Time']).dt.minute

test_set.drop(labels='Arrival_Time', axis = 1, inplace= True)

test_set.head()
training_set['Airline'].value_counts()
labels = (training_set.Airline.unique())

values = training_set.Airline.value_counts()



trace = go.Pie(labels=labels, values=values)



iplot([trace])
import plotly.express as px

#df = px.data.training_data()

fig = px.box(training_set, x="Airline", y="Price", color="Airline", notched=True)

fig.show()
# Convert Categorical data to Numeric using one-hot encoding as Airline is nominal data

Airline = training_set[["Airline"]]



Airline = pd.get_dummies(Airline, drop_first= True)



Airline.head()
training_set['Source'].value_counts()
fig = px.box(training_set, x="Source", y="Price", color="Source", notched=True)

fig.show()
Source = training_set[["Source"]]



Source = pd.get_dummies(Source, drop_first= True)



Source.head()
training_set['Destination'].value_counts()
fig = px.box(training_set, x="Destination", y="Price", color="Destination", notched=True)

fig.show()
Destination = training_set[["Destination"]]



Destination = pd.get_dummies(Destination, drop_first= True)



Destination.head()
training_set['Route'].value_counts()
training_set["Total_Stops"].value_counts()
sns.catplot(y = "Price", x = "Total_Stops", data = training_set.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)

plt.show()
#

training_set.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


training_set.head()
fig = px.box(training_set, x="Journey_Month", y="Price", color="Journey_Month")

fig.show()
data_train = pd.concat([training_set, Airline, Source, Destination], axis=1)
data_train.isna().any()
data_train.head()
data_train['Additional_Info'].value_counts()
data_train.drop(["Airline", "Source", "Destination", "Route", "Additional_Info"], axis=1, inplace = True)
data_train.columns
# Categorical data to Numeric for test_set



print(test_set["Airline"].value_counts())

Airline = pd.get_dummies(test_set["Airline"], drop_first= True)



print()



print(test_set["Source"].value_counts())

Source = pd.get_dummies(test_set["Source"], drop_first= True)



print()



print(test_set["Destination"].value_counts())

Destination = pd.get_dummies(test_set["Destination"], drop_first = True)



# Additional_Info contains almost 80% no_info

# Route and Total_Stops are related to each other

test_set.drop(["Route", "Additional_Info"], axis = 1, inplace = True)



# Replacing Total_Stops

test_set.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)



# Concatenate dataframe --> test_data + Airline + Source + Destination

data_test = pd.concat([test_set, Airline, Source, Destination], axis = 1)



data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

data_test.head()

data_test.shape
data_test.columns
data_train.shape
data_train.isna().any()
null_in_total_stops = data_train[data_train['Total_Stops'].isnull()]

null_in_total_stops
#delete_row = data_train[data_train['Total_Stops'].isnull()].index

#print(delete_row)

#data_train = data_train.drop(delete_row)
#Checking for null values

print(data_train.isnull().values.any())
data_train.columns
y = data_train.iloc[:, 1]

y
X = data_train.loc[:, ['Total_Stops', 'Journey_Day', 'Journey_Month',

       'Duration_hours', 'Duration_minutes', 'Dep_hour', 'Dep_min',

       'Arrival_hour', 'Arrival_min', 'Airline_Air India', 'Airline_GoAir',

       'Airline_IndiGo', 'Airline_Jet Airways', 'Airline_Jet Airways Business',

       'Airline_Multiple carriers',

       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',

       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',

       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',

       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',

       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()
X.isnull().sum()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
X_test
# Feature Scaling So that data in all the columns are to the same scale

#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#X_train_scalar = sc.fit_transform(X_train)

#X_test_scalar = sc.fit_transform(X_test)

#X_test_scalar
RandomForestModel = RandomForestRegressor()

RandomForestModel.fit(X_train,y_train)
y_pred = RandomForestModel.predict(X_test)

y_pred
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("RMSE", rmse)

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)
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

rf_random = RandomizedSearchCV(estimator = RandomForestModel, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
y_pred_tune = rf_random.predict(X_test)

X_test
mse = mean_squared_error(y_test, y_pred_tune)

rmse = np.sqrt(mse)

print("RMSE", rmse)

r2 = r2_score(y_test, y_pred_tune)

print("R2 Score:", r2)
import pickle

# Writing different model files to file

with open( 'rfModelPrediction.pkl', 'wb') as file:

    pickle.dump(rf_random,file)
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("RMSE", rmse)

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)
from xgboost import XGBRegressor



xgb_model = XGBRegressor()

xgb_model.fit(X_train, y_train)



y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("RMSE", rmse)

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)

import pickle

# Writing different model files to file

with open( 'xgbmodelPrediction.pkl', 'wb') as file:

    pickle.dump(xgb_model,file)

from sklearn.model_selection import GridSearchCV
param_grid={

    'learning_rate':[1,0.3,0.1,0.01],

    'n_estimators':[50,300,800],

    'max_depth':[3,5,10]

}
grid= GridSearchCV(XGBRegressor(objective='reg:squarederror'), param_grid, verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
import numpy as np

new_model=XGBRegressor(learning_rate= 0.1, max_depth= 10, n_estimators= 50, n_jobs = -1)

new_model.fit(X_train, y_train)

# make prediction

y_pred = new_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

#rmse = np.sqrt(mse)

print("RMSE", rmse)

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)
import pickle

# Writing different model files to file

with open( 'modelForPrediction.pkl', 'wb') as f:

    pickle.dump(new_model,f)

    
