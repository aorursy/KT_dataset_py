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
train=pd.read_excel("/kaggle/input/flight-fare-prediction-mh/Data_Train.xlsx")
train.head()
train.info()
train=train.dropna()
train.isnull().sum()
#convert day from object datatype to datetime object
train["Date_of_Journey"]
train["day_of_journey"]=pd.to_datetime(train.Date_of_Journey,format="%d/%m/%Y").dt.day
train["month_of_journey"]=pd.to_datetime(train.Date_of_Journey,format="%d/%m/%Y").dt.month
train.drop("Date_of_Journey",axis=1,inplace=True)
train.head()
#convert depart time into datetime format
train["Dep_Hour"]=pd.to_datetime(train["Dep_Time"]).dt.hour
train["Dep_Minute"]=pd.to_datetime(train["Dep_Time"]).dt.minute
train.drop("Dep_Time",axis=1,inplace=True)
train
# Arrival time is when the plane pulls up to the gate.
# Similar to Date_of_Journey we can extract values from Arrival_Time

# Extracting Hours
train["Arrival_hour"] = pd.to_datetime(train.Arrival_Time).dt.hour

# Extracting Minutes
train["Arrival_min"] = pd.to_datetime(train.Arrival_Time).dt.minute

# Now we can drop Arrival_Time as it is of no use
train.drop(["Arrival_Time"], axis = 1, inplace = True)
train
#splitting the duration into hours and minutes
duration=list(train["Duration"])
print(len(duration[0].split()))
for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if "h" in duration[i]:
            duration[i]=duration[i].strip()+" 0m"
        else:
            duration[i]="0h "+duration[i]
duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))
    





train["Duration_hours"]=duration_hours
train["Duration_mins"]=duration_mins
train.drop("Duration",axis=1,inplace=True)
train["Airline"].value_counts()
#plotting airline against price using box plot
import seaborn as sns
sns.catplot(y="Price",x="Airline",data=train.sort_values("Price",ascending=False),kind="boxen",height=6,aspect=3)

#one hot encoding of Airline (Nominal data)
airlines=train["Airline"]
airlines=pd.get_dummies(airlines,drop_first=True)



airlines


# Source vs Price

sns.catplot(y = "Price", x = "Source", data = train.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)



source = train["Source"]

source = pd.get_dummies(source, drop_first= True)
destination = train["Destination"]

destination = pd.get_dummies(destination, drop_first = True)

destination.head()


train["Additional_Info"].value_counts()
#most of the values are no_info
train.drop("Additional_Info",axis=1,inplace=True)
train["Route"]
train["Total_Stops"]
train.drop("Route",axis=1,inplace=True)
stops=list(train["Total_Stops"])
for i in range(len(stops)):
    if stops[i]=="non-stop":
        stops[i]=0
    else:
        stops[i]=stops[i].split()[0]

train["Num_stops"]=stops
train.head()
train.drop("Total_Stops",axis=1,inplace=True)
#concatenate train dataframe with one-hot encoded data
train=pd.concat([train,airlines,destination,source],axis=1)
train.head()
train.drop(["Airline","Source","Destination"],axis=1,inplace=True)
train.head()
test_data=pd.read_excel("/kaggle/input/flight-fare-prediction-mh/Test_set.xlsx")
test_data.info()
test_data.isnull().sum()
test_data.dropna(inplace=True)
# Date_of_Journey
test_data["day_of_journey"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["month_of_journey"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_Hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_Min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
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

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)
airlines=test_data["Airline"]
airlines=pd.get_dummies(airlines,drop_first=True)


source = test_data["Source"]

source = pd.get_dummies(source, drop_first= True)
destination = test_data["Destination"]

destination = pd.get_dummies(destination, drop_first = True)

destination.head()
stops=list(test_data["Total_Stops"])
for i in range(len(stops)):
    if stops[i]=="non-stop":
        stops[i]=0
    else:
        stops[i]=stops[i].split()[0]

test_data["Num_stops"]=stops
test_data.head()
test_data=pd.concat([test_data,airlines,destination,source],axis=1)
test_data.drop(["Airline","Source","Destination","Route","Total_Stops"],axis=1,inplace=True)
train.shape
test_data.shape
#heatmap
import matplotlib.pyplot as plt
plt.figure(figsize = (18,18))
sns.heatmap(train.corr(), annot = True, cmap = "RdYlGn")

plt.show()
#separate dataset into dependent and independent features
X=train.drop("Price",axis=1)
X=X.iloc[:,:].values
Y=train["Price"]
Y=Y.iloc[:].values
Y
# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, Y)


print(selection.feature_importances_)


#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=train.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
reg_rf.score(X_train, y_train)
reg_rf.score(X_test, y_test)
sns.distplot(y_test-y_pred)
plt.show()
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
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
prediction=rf_random.predict(X_test)
plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


#save the model to reuse it again
import pickle
file=open("flight_rf.pkl","wb")
pickle.dump(rf_random,file)
model=open("flight_rf.pkl","rb")
forest=pickle.load(model)
ypred=forest.predict(X_test)
metrics.r2_score(y_test,ypred)