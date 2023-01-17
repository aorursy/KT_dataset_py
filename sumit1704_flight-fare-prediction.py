# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
train_data = pd.read_excel("../input/flight-fare-prediction-mh/Data_Train.xlsx")
train_data.columns, train_data.shape
train_data.dtypes
train_data.describe()
# checking if the data has some missing values
train_data.isnull().sum()
# dropping the missing data from the original data
train_data.dropna(inplace = True)
train_data.isnull().sum()
sns.distplot(train_data['Price']);
sns.distplot(np.log(train_data['Price']));
np.log(train_data['Price']).describe()
train_data['Price'].describe()
train_data['Date_of_Journey'] = pd.to_datetime(train_data['Date_of_Journey'],format= "%d/%m/%Y")
# getting the day pf the journey
train_data['Day_of_journey'] = train_data['Date_of_Journey'].dt.day_name()
train_data['Month_of_journey'] = train_data['Date_of_Journey'].dt.month_name()

#removing the date of journey column as all the features are extracted and it's of no use.
train_data.drop(['Date_of_Journey'], axis = 1 , inplace = True)
train_data.head()
train_data['Day_of_journey'].value_counts(), train_data['Month_of_journey'].value_counts()
sns.countplot(x = 'Month_of_journey', data = train_data);
#it can be seen that the month with the highest number of flights is june and the least one is April.
sns.countplot(x = 'Day_of_journey', data = train_data);
#we can see that most of the flights are on thursday and the least are on Sunday.
train_data.groupby(['Month_of_journey']).mean()
#it can be seen that the prices are the highest in Januart while the price for a flight in april is the least
#Comparing the mean of the price according to the day
#we can say that the highest prices are on thursday while lowest being on friday.
train_data.groupby(['Day_of_journey']).mean()

#extracting hours and minutes and removing the Dep_Time after feature extraction

train_data['departure_hour'] = pd.to_datetime(train_data["Dep_Time"]).dt.hour
train_data['departure_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute

train_data.drop(['Dep_Time'], axis = 1, inplace = True)
train_data.head()
# extracting features from arrival time and then removing it
train_data['arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
train_data['arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute
#dropping
train_data.drop(['Arrival_Time'], axis = 1, inplace = True)
'h' in '20h'
#using pandas to process the Duration column for further analysis as it is in string format originally

def get_hr_min(return_h, o):
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
        
    return h if return_h else m
from functools import partial 
f = partial(get_hr_min, True)
train_data['duration_hr'] = train_data.Duration.map(f)
f = partial(get_hr_min, False)
train_data['duration_min'] = train_data.Duration.map(f)
'2h 50m'.split()
train_data.head()
train_data.drop(['Duration'], axis = 1, inplace = True)
train_data.head()
#Handling Categorical Data
train_data.dtypes
train_data["Airline"].value_counts()
sns.catplot(y = 'Price', x = 'Airline',data= train_data.sort_values("Price", ascending = False), kind = "boxen",
            height = 8, aspect = 2)
plt.tight_layout;
#we can see that the Jet airways business has the highest price
#using onehotcoding to change the categorical data into numerical
Airline = train_data[['Airline']]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()  
#checking source variable

train_data.Source.value_counts()
sns.catplot(y = 'Price', x = 'Source', data = train_data.sort_values("Price", ascending= False), kind = "boxen", height=4, aspect = 2)
plt.tight_layout;
#Since Source is also categorical data we will use onehotencoding to change it to numerical

Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()
train_data.Destination.value_counts()
#again performing onehotcoding to convert destination into numerical data
Destination = train_data[['Destination']]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()
train_data.Additional_Info.value_counts()
train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
train_data.head()
train_data[:10]
train_data.Total_Stops.value_counts()
#using label encoding for Total Stops
train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops":3, "4 stops": 4}, inplace = True)
train_data
#one hot encoding for journey day and journey month

journey_day = pd.get_dummies(train_data['Day_of_journey'], drop_first = True)
journey_month = pd.get_dummies(train_data['Month_of_journey'], drop_first = True)

#removing the Airline Source and Destination columns as the information has already been extracted from them

train_data.drop(["Airline", "Source", "Destination","Day_of_journey", "Month_of_journey"], inplace = True, axis = 1)
data_train = pd.concat([train_data, Airline, Source, Destination, journey_day, journey_month], axis = 1)
data_train.head(), data_train.shape
data_train.to_excel("Flight.xls")
test_data = pd.read_excel("../input/flight-fare-prediction-mh/Test_set.xlsx")
print(test_data.head(), test_data.shape)
test_data.columns, test_data.dtypes
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
# extracting hours and minutes from duration column
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hr = []
duration_min = []

for i in range(len(duration)):
    duration_hr.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_min.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
    
# Adding Duration column to test set
test_data["Duration_hr"] = duration_hr
test_data["Duration_min"] = duration_min
test_data.drop(["Duration"], axis = 1, inplace = True)
# dropping Route and additional information columns
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
#label encoding for total_stops

test_data.replace({"non-stop": 0,"1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

#one hot encoding for Airline, Source and Destination
print("Airline")
print("-"*75)
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print("Source")
print("-"*75)
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print("Destination")
print("-"*75)
Destination = pd.get_dummies(test_data["Destination"], drop_first= True)

test_data.drop(["Airline","Source","Destination"], axis = 1, inplace = True)
test_data.head()

#one hot encoding for journey day and month for test data

journey_day = pd.get_dummies(test_data["Day_of_journey"], drop_first= True)
journey_month = pd.get_dummies(test_data["Month_of_journey"], drop_first= True)

#removing the extra columns
test_data.drop(["Day_of_journey","Month_of_journey"], axis = 1, inplace = True)

print("shape of test data", test_data.shape)
#joining the data after encoding for further processing
data_test = pd.concat([test_data, Airline, Source, Destination, journey_day, journey_month], axis = 1)
print("shape of test data", data_test.shape)
data_test.to_excel("Flight_test.xls")
data_train.columns
X = data_train.loc[:,['Total_Stops',
    'departure_hour', 'departure_min',
       'arrival_hour', 'arrival_min', 'duration_hr', 'duration_min',
       'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi', 'Monday', 'Saturday',
       'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'June', 'March', 'May']]

X.head()


y = data_train.iloc[:,1]
y.head()
#using heatmap to find the correlation of the variables

plt.figure(figsize = (15,10))
sns.heatmap(train_data.corr(), annot = True, cmap= "YlGnBu")

plt.tight_layout()

#Important feature using Extratreesregressor
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,y)

print(selection.feature_importances_)

#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

#fitting the model
from sklearn.ensemble import RandomForestRegressor
model_1 = RandomForestRegressor()
model_1.fit(X_train, y_train)
#making prediction on test data
y_pred = model_1.predict(X_test)
model_1.score(X_train, y_train)
model_1.score(X_test, y_test)
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
print("R-squared score:", metrics.r2_score(y_test, y_pred))