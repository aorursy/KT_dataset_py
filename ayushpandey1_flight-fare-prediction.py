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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
#importing dataset and displaying
train_df=pd.read_excel(r'../input/flight-fare-prediction-mh/Data_Train.xlsx')
train_df
#displaying the max columns
pd.set_option('display.max_columns',None)
train_df.head()
#displaying the info of each column
train_df.info()
#unique values in the duration column of dataset
train_df['Duration'].value_counts()
#dropping the NAN value null of the dataset
train_df.dropna(inplace=True)
train_df.shape
#showing the sum of all the null values  in each column of data
train_df.isnull().sum()
#extracting the day of journey and month of journey from the Date of journey column
#as both day and month are required and model will not understand string values

train_df["Journe_day"] = pd.to_datetime(train_df['Date_of_Journey'],format="%d/%m/%Y").dt.day
train_df["Journey_month"] = pd.to_datetime(train_df['Date_of_Journey'], format= "%d/%m/%Y").dt.month
train_df.head()
train_df.drop(["Date_of_Journey"],axis=1,inplace=True)
## Departure time is when a plane leaves the gate. 
# Similar to Date_of_Journey we can extract values from Dep_Time

#Extracting hours
train_df["Dep_hour"] = pd.to_datetime(train_df["Dep_Time"]).dt.hour

#Extracting ,minutes
train_df["Dep_min"]=pd.to_datetime(train_df["Dep_Time"]).dt.minute

#Now we can drop Dep_Time as it is of no use
train_df.drop(['Dep_Time'],axis=1, inplace=True)
train_df.head()
# Arrival time is when the plane pulls up to the gate.
# Similar to Date_of_Journey we can extract values from Arrival_Time


#Extracting hours
train_df["Arrival_hour"] = pd.to_datetime(train_df.Arrival_Time).dt.hour

#Extracting Mintes
train_df["Arrival_min"] = pd.to_datetime(train_df.Arrival_Time).dt.minute

# Now we can drop Arrival_Time as it is of no use
train_df.drop(["Arrival_Time"],axis=1, inplace=True)
train_df.head()
# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(train_df["Duration"])

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

train_df["Duration_hours"] = duration_hours
train_df["Duration_mins"] = duration_mins
train_df.drop(["Duration"], axis = 1, inplace = True)

train_df.head()
#count the number of times each category is there
train_df['Airline'].value_counts()
# From graph we can see that Jet Airways Business have the highest Price.
# Apart from the first Airline almost all are having similar median

# Airline vs Price
sns.catplot(y="Price", x="Airline", data=train_df.sort_values("Price",ascending=False),kind="boxen" ,height=5,aspect=3)
plt.show()

# As Airline is Nominal Category data we will perform OneHotEncoding
#we cannot differentiate between categories of Airline
#we will take get_dummies which is part of OneHotEncoding

Airline = train_df[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)  # we will drop first feature which is not required

Airline.head()
#counting the number of times each category in Source happens
train_df["Source"].value_counts()
#category plot
#Source vs Price
sns.catplot(x = "Source",y = "Price", data = train_df.sort_values("Price", ascending = False), kind="boxen", height = 5, aspect = 3)
plt.show()

# As Source is Nominal Categorical data we will perform OneHotEncoding
#we cannot differentiate between categories of Source as it is categorical data
#we will perform get_dummies which is part of OneHotEncoding
Source = train_df[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()
train_df["Destination"].value_counts()
# Destination vs Price
sns.catplot(y = "Price", x = "Destination", data = train_df.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()

#As Destination is Nominal Categorical data we will perform OneHotEncoding
Destination = train_df[['Destination']]


Destination = pd.get_dummies(Destination , drop_first = True)

Destination.head()

#Route and Total Stops doing same thing 
#From Route we will come to know the number of stops
train_df["Route"]
#We will drop Route and Additional_Info as Additional_Info contains
   # 80% no_info


train_df.drop(["Route" , "Additional_Info"], axis=1, inplace=True)
train_df.head()
#Now total_Stops again is a Categorical Fature

train_df["Total_Stops"].value_counts()
#So we came to know1 stop occurs 5625 times
#2 stops occur 1520 times
#As this is case of Ordinal Categorical data type we use LabelEncoder
# Here values are assigned with corresponding types

train_df.replace({"non-stop":0 , "1 stop":1 , "2 stops":2 ,"3 stops":3 ,"4 stops":4} ,inplace=True)
train_df.head()
# Concatenate dataframe --> train_df + Airline + Source + Destination


df_train = pd.concat([train_df ,Airline,Source,Destination] ,axis=1)
#checking the first five rows of new dataframe
df_train.head()
# Drop Airline Source and Destination as we have already converted them into OneHotEncoding

df_train.drop(["Airline" , "Source", "Destination"] ,  axis=1 , inplace =True)
df_train.head()

# showing the shape of new dataframe

df_train.shape
# As we are seeing we are not given the PRICE in test data which we have to predict
test_df=pd.read_excel(r'../input/flight-fare-prediction-mh/Test_set.xlsx')
test_df
# Preprocessing steps

print("Test data info")
print("*" *75)
print(test_df.info())
print("Null values :")
print("*"*75)
test_df.dropna(inplace = True)
print(test_df.isnull().sum())

# EDA

# Date_of_Journey
test_df["Journey_day"] = pd.to_datetime(test_df.Date_of_Journey, format="%d/%m/%Y").dt.day
test_df["Journey_month"] = pd.to_datetime(test_df["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_df.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_df["Dep_hour"] = pd.to_datetime(test_df["Dep_Time"]).dt.hour
test_df["Dep_min"] = pd.to_datetime(test_df["Dep_Time"]).dt.minute
test_df.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_df["Arrival_hour"] = pd.to_datetime(test_df.Arrival_Time).dt.hour
test_df["Arrival_min"] = pd.to_datetime(test_df.Arrival_Time).dt.minute
test_df.drop(["Arrival_Time"], axis = 1, inplace = True)

duration = list(test_df["Duration"])

for i in range(len(duration)):
    if (len(duration[i].split())!=2):                       # Chech if duration contains only hours and minutes
        if "h" in duration[i]:
           duration[i] = duration[i].strip() +"0m"          #Adds 0 minutes
        else:
           duration[i] = "0h" + duration[i]                 # adds 0 hours

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(str(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(str(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
    
    
# Adding Duration column to test set
test_df["Duration_hours"] = duration_hours
test_df["Duration_mins"] = duration_mins
test_df.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_df["Airline"].value_counts())
Airline = pd.get_dummies(test_df["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_df["Source"].value_counts())
Source = pd.get_dummies(test_df["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_df["Destination"].value_counts())
Destination = pd.get_dummies(test_df["Destination"], drop_first = True)


# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
df_test = pd.concat([test_df, Airline, Source, Destination], axis = 1)

df_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", df_test.shape)
df_test.head()
df_train.shape
df_train.columns
# Extracting the independent variables 

X = df_train[['Total_Stops', 'Journe_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()

#Extracting the dependent Variable

Y=df_train.iloc [:,1]
Y.head()
# Finds correlation between Independent and dependent attributes
corr = train_df.corr()
plt.figure(figsize = (18,18))
sns.heatmap(corr , annot = True, cmap = "RdYlGn")

plt.show()
# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,Y)
print(selection.feature_importances_)
# plot graph of Feature importances for better visualization

plt.figure(figsize = (12,8))
feat_imp = pd.Series(selection.feature_importances_ , index= X.columns)
feat_imp.nlargest(20).plot(kind ='barh')
plt.show()

#splitting the dependent ad independent variable into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
#Fitting the RandomForest Model on Xtrain and Ytrain
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, Y_train)
#predicting the value on X_test

Y_pred = reg_rf.predict(X_test)

reg_rf.score(X_train, Y_train)
reg_rf.score(X_test, Y_test)
#plotting the distribution plot and we find the Gaussian plot

sns.distplot(Y_test-Y_pred)
plt.show()
plt.scatter(Y_test, Y_pred, alpha = 0.5)
plt.xlabel("Y_test")
plt.ylabel("Y_pred")
plt.show()
from sklearn import metrics
#finding the Errors 

print('MAE:', metrics.mean_absolute_error(Y_test, Y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
metrics.r2_score(Y_test, Y_pred)
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
rf_random = RandomizedSearchCV( estimator = reg_rf , param_distributions= random_grid , scoring='neg_mean_squared_error',
                               n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train , Y_train)
rf_random.best_params_
prediction = rf_random.predict(X_test)
plt.figure(figsize = (8,8))
sns.distplot(Y_test-prediction)
plt.show()

plt.figure(figsize = (8,8))
plt.scatter(Y_test, prediction, alpha = 0.5)
plt.xlabel("Y_test")
plt.ylabel("Y_pred")
plt.show()
print('MAE:', metrics.mean_absolute_error(Y_test, prediction))
print('MSE:', metrics.mean_squared_error(Y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, prediction)))
print("R2 Score of Our Model is : ")
print()
metrics.r2_score(Y_test, prediction)
import pickle
my_file= open('flight_fa.pkl' , 'wb')
pickle.dump(rf_random , my_file)
model = open('flight_fa.pkl' ,'rb')
forest =pickle.load(model)
pred = forest.predict(X_test)
metrics.r2_score(Y_test , pred)
