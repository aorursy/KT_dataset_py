import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
flight_2019 = pd.read_csv("/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv")
flight_2020 = pd.read_csv("/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv")
## PROBLEM STATEMENT:

# Based on given data predict future flight delays for the month of January
flight_2019.head()
flight_2020.head()
# Shape of datasets
flight_2019.shape
flight_2020.shape
# Info on datasets:
flight_2019.info()
flight_2020.info()
# Describe on datasets:
flight_2019.describe().transpose()
flight_2020.describe().transpose()
# Dataset columns:
flight_2019.columns
flight_2020.columns
# Both datasets have the same columns 
# Checking for duplicate entries in dataset
flight_2019.duplicated(keep = "first").value_counts()
flight_2020.duplicated(keep = "first").value_counts()

# Data Cleaning
# Adding new "YEAR" parameter to the two datasets:
flight_2019["YEAR"] = 2019
flight_2020["YEAR"] = 2020
# Adding new "MONTH" parameter to the two datasets:
# "MONTH" will be 1 as data is for the month of "JANUARY"
flight_2019["MONTH"] = 1
flight_2020["MONTH"] = 1
# Thus, now we have "YEAR" , "MONTH" , "DAY_OF_MONTH", "DAY_OF_WEEK" : dates of the flight
#  Renaming "DAY_OF_MONTH" as "DAY"
flight_2019.rename(columns = {"DAY_OF_MONTH" : "DAY"}, inplace= True)
flight_2020.rename(columns = {"DAY_OF_MONTH" : "DAY"}, inplace= True)


flight_2019['DATE'] = pd.to_datetime(flight_2019[['YEAR','MONTH', 'DAY']])
flight_2020['DATE'] = pd.to_datetime(flight_2020[['YEAR','MONTH', 'DAY']])
flight_2019["DATE"].unique()
flight_2020["DATE"].unique()
flight_2019.columns
# Handling "DEP_TIME" and "ARR_TIME" in both datasets:

# They are mentioned in "float64" format 
import datetime
def format_time(t):
    if pd.isnull(t):
        return np.nan
    else:
        if t == 2400: t = 0
        t = "{0:04d}".format(int(t))
        result = datetime.time(int(t[0:2]), int(t[2:4]))
        return result
# Change format of arrival time and departure time in both datasets :
flight_2019['DEP_TIME'] = flight_2019['DEP_TIME'].apply(format_time)
flight_2020['DEP_TIME'] = flight_2020['DEP_TIME'].apply(format_time)
flight_2019['ARR_TIME'] = flight_2019['ARR_TIME'].apply(format_time)
flight_2020['ARR_TIME'] = flight_2020['ARR_TIME'].apply(format_time)

# columns in changed format
flight_2019["DEP_TIME"]
flight_2019["ARR_TIME"]
flight_2020["DEP_TIME"]
flight_2020["ARR_TIME"]
flight_2019.head()
flight_2020.head()

## Concatenate the two datasets:

flight = pd.concat([flight_2019, flight_2020])
flight.shape
# info on flight data
flight.info()
# describe on flight data
flight.describe()
# check null values
flight.isna().sum()
# null values in percentage 

((flight.isna().sum()/len(flight))*100).round(2)
## Dropping "Unnamed: 21" as all its entries are empty
flight = flight.drop(columns = ["Unnamed: 21"])
# Dropping rows for which columns have very less percentage of null values ( < 3%)
flight = flight.dropna()
## Now our data has no null entries
flight.nunique()
# find and remove nunique == 1
flight.nunique()[flight.nunique() == 1].index.to_list()
# Two constant feature columns found : "CANCELLED" and "DIVERTED"
flight = flight.drop(columns = ["CANCELLED", "DIVERTED"])
flight.columns.size

# value_counts() for columns[0:10]
i=0
for col in flight.columns[:10]:
  print("Columns no is: " + str(i))
  i = i+1
  print("Column is : " + col)
  print(flight[col].value_counts())
  print()
  print()
# value_counts() for columns[10:19]
for col in flight.columns[10:]:
  print("Column no is: " + str(i))
  i = i+1
  print("Column is : " + col)
  print(flight[col].value_counts())
  print()
  print()
## Here, from above we can conclude the following:

# "OP_CARRIER" and "OP_UNIQUE_CARRIER" refect the same category of information
# -- one of them can be dropped
# "OP_CARRIER" and "OP_CARRIER_AIRLINE_ID" also represent the same information
# -- all categories and their counts are same for both columns

## "ORIGIN_AIRPORT_ID" and "ORIGIN" also represent same information 
# It is inherently categorical in nature -- hence, we drop "ORIGIN_AIRPORT_ID"

## "DEST_AIRPORT_ID" and "DEST" also represent same information
# It is also inherently categorical in nature -- hence, we drop "DEST_AIRPORT_ID"
flight = flight.drop(columns = ["OP_UNIQUE_CARRIER", "OP_CARRIER_AIRLINE_ID", "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID"])

flight.columns
## "MONTH" , "YEAR", "DAY" are not required

flight = flight.drop(columns = ["MONTH", "YEAR", "DAY"])
# Renaming :

flight.rename(columns = {"DEP_DEL15" : "DEP_DELAY"}, inplace= True)
flight.rename(columns = {"ARR_DEL15" : "ARR_DELAY"}, inplace= True)

## Removing "ARR_TIME" and "ARR_DELAY" as "DEP_DELAY" is considered as TARGET variable
flight = flight.drop(columns= ["ARR_DELAY", "ARR_TIME"])

## DATA VISUALIZATION:

## Analysis of target variable

flight["DEP_DELAY"].value_counts()
# Plot to visualize the target variable in the dataset:

sns.set(style="darkgrid")

plt.figure(figsize=(10,6))
ax = sns.countplot(x="DEP_DELAY", data=flight)
plt.title('Distribution of target variable')
plt.xlabel('0 - NO DEPARTURE DELAY  |  1 -DEPARTURE DELAY')
plt.ylabel('Frequency')

for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
# Percentage of True/False for Target Variable

print("Percentage of No Departure Delay")
print(((flight["DEP_DELAY"].value_counts()[0]/flight.shape[0])*100).round(2))
print()
print("Percentage of Departure Delay")
print(((flight["DEP_DELAY"].value_counts()[1]/flight.shape[0])*100).round(2))
# Pie chart for the TARGET variable "DEP_DELAY"

plt.figure(figsize = (7, 7))
flight["DEP_DELAY"].value_counts().plot.pie()
plt.show()

# 0 -> NO DELAY 
# 1 -> DELAY
flight.corr()
# Plotting the correaltion data heatmap

plt.figure(figsize = (4,4))
sns.heatmap(flight.corr())
plt.show()

# Model Building :
# Dropping columns 
flight = flight.drop(columns = ["TAIL_NUM", "ORIGIN_AIRPORT_SEQ_ID", "DEST_AIRPORT_SEQ_ID"])
flight.head()
## We will require the day_of_month column only out of the date column , as year is in two categories only
# and month is january "1" for all rows hence, that is not a factor

flight["DAY_OF_MONTH"] = flight["DATE"].dt.day
## Drop "DATE" feature:
flight = flight.drop(columns = "DATE")
flight.head()
## Moving "DEP_DELAY" column and making it the last colunm in the dataset:
cols_at_end = ["DEP_DELAY"]
flight_data = flight[[c for c in flight if c not in cols_at_end] 
        + [c for c in cols_at_end if c in flight]]
flight_data.head()
import warnings
warnings.filterwarnings("ignore")
## Selecting the cyclic features in the datasets: They need to be encoded separately
def encode(data, col, max_val):
    data.loc[:, col + '_SIN'] = np.sin(2 * np.pi * data[col]/max_val)
    data.loc[:, col + '_COS'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

encode( flight_data, "DAY_OF_WEEK", 7)
encode( flight_data, "DAY_OF_MONTH", 30)
flight_data.columns
## Now, we can drop the original "DAY_OF_WEEK" and "DAY_OF_MONTH" columns from the dataset
flight_data = flight_data.drop(columns = ["DAY_OF_WEEK", "DAY_OF_MONTH"])
flight_data.dtypes
flight_data["DEP_TIME"]
## We have departure time in 24 hour format 
## -- In order to apply cyclic encoding , we will have to convert it into minutes past midnight
## -- and then apply cyclic encoding to the dataset
# Extract hour from the "DEP_TIME" using str
flight_data["DEP_TIME"].astype(str).str[0:2]
# Extract minutes from the "DEP_TIME" using str
flight_data["DEP_TIME"].astype(str).str[3:5]
# "DEP_TIME" converted into minutes past midnight

((flight_data["DEP_TIME"].astype(str).str[0:2]).astype(int))*60 + (flight_data["DEP_TIME"].astype(str).str[3:5].astype(int))
dep_time_in_minutes = ((flight_data["DEP_TIME"].astype(str).str[0:2]).astype(int))*60 + (flight_data["DEP_TIME"].astype(str).str[3:5].astype(int))
def add_new_col(data, col, new_col):
    data.loc[:, col + '_MINUTES'] = new_col
    return data
## Creating "DEP_TIME_MINUTES" in terms of minutes_data:
add_new_col(flight_data, "DEP_TIME", dep_time_in_minutes)
max_time_in_a_day = 24*60

encode( flight_data, "DEP_TIME_MINUTES", max_time_in_a_day)

flight_data.columns
## Columns added with sin and cos parameters w.r.t "DEP_TIME_MINUTES" column
## Now, we can drop "DEP_TIME" column and "DEP_TIME_MINUTES"
flight_data = flight_data.drop(columns = ["DEP_TIME", "DEP_TIME_MINUTES"])
## Dtypes of columns present in the dataset at this point
flight_data.dtypes
## Applying LabelEncoding on "ORIGIN" and "DEST" columns:

flight_data["ORIGIN"] = flight_data["ORIGIN"].astype('category')
flight_data["DEST"] = flight_data["DEST"].astype('category')

flight_data["ORIGIN_ENCODE"] = flight_data["ORIGIN"].cat.codes
flight_data["DEST_ENCODE"] = flight_data["DEST"].cat.codes

## Drop "ORIGIN" and "DEST" columns 
flight_data = flight_data.drop(columns = ["ORIGIN", "DEST"])
# One Hot encoding
flight_data = pd.get_dummies(data = flight_data, columns = ["OP_CARRIER", "DEP_TIME_BLK"], drop_first = True)
flight_data.head()
# Moving "DEP_DELAY" column and making it the last colunm in the dataset:
cols_at_end = ["DEP_DELAY"]
flight_data = flight_data[[c for c in flight_data if c not in cols_at_end] 
        + [c for c in cols_at_end if c in flight_data]]
flight_data.dtypes
# Loading X and y parameters for our model building

X = flight_data.iloc[:, :-1].values
y = flight_data.iloc[:, -1].values
# Split data 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# find index of columns to be scaled 
print(flight_data.columns.get_loc("OP_CARRIER_FL_NUM"))
print(flight_data.columns.get_loc("DISTANCE"))
print(flight_data.columns.get_loc("ORIGIN_ENCODE"))
print(flight_data.columns.get_loc("DEST_ENCODE"))

# Applying Feature Scaling 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, [0, 1, 8, 9]] = sc.fit_transform(X_train[:, [0, 1, 8, 9]])
X_test[:, [0, 1, 8, 9]] = sc.transform(X_test[:, [0, 1, 8, 9]])

# Model 1: Logistic Regression

from sklearn.linear_model import LogisticRegression

log_classifier = LogisticRegression(random_state = 0)
log_classifier.fit(X_train, y_train)
# Predicting the test set result : Logistic Classifier

y_pred = log_classifier.predict(X_test)
# Making the confusion matrix 

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : Logistic Regression")
print(cm)
print()
print("Accuracy Score : Logistic Regression")
print( accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# Model 2: Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
# Predicting the test set result : Naive Bayes

y_pred_nb = nb_classifier.predict(X_test)
# Making the confusion matrix : Naive Bayes

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_nb)
print("Confusion Matrix : Naive Bayes ")
print(cm)
print()
print("Accuracy Score : Naive Bayes ")
print( accuracy_score(y_test, y_pred_nb))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_nb))
# Model 3: DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)
# Predicting the test set result : Decision Tree Classifier

y_pred_dt = dt_classifier.predict(X_test)
# Making the confusion matrix : Decision Tree Classifier

cm = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix : Decision Tree Classifier ")
print(cm)
print()
print("Accuracy Score : Decision Tree Classifier ")
print( accuracy_score(y_test, y_pred_dt))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_dt))

## Applying K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_decision_tree = cross_val_score(estimator = dt_classifier, X = X_train, y = y_train, cv = 10)
print("Accuracies mean for Decision tree is : ")
print(accuracies_decision_tree.mean())
print()
print("Standard Deviation for Decision Tree accuracies is :")
print(accuracies_decision_tree.std())
# Model 4: Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)
# Predicting the test set result : Random Forest Classifier

y_pred_rf = rf_classifier.predict(X_test)
# Making the confusion matrix : Random Forest Classifier

cm = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix : Random Forest Classifier ")
print(cm)
print()
print("Accuracy Score : Random Forest Classifier ")
print( accuracy_score(y_test, y_pred_rf))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rf))
## Applying K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_random_forest = cross_val_score(estimator = rf_classifier, X = X_train, y = y_train, cv = 10)
print("Accuracies mean for Random Forest is : ")
print(accuracies_random_forest.mean())
print()
print("Standard Deviation for Random Forest accuracies is :")
print(accuracies_random_forest.std())
