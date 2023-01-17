#----------------------EDA Librarries------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn
from sklearn.linear_model import LinearRegression
import  statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
train = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')
pd.set_option('display.max_columns', None)
train.head()
train.info()
train.shape
mn.matrix(train)
train.isnull().sum()
train.dropna(axis = 0,inplace = True)
train.isnull().sum()
train['Date_of_Journey'] = train['Date_of_Journey'].astype('datetime64[ns]')

train['Journey_Day'] = train['Date_of_Journey'].dt.day
train['Journey_Month'] = train['Date_of_Journey'].dt.month
train.head()
train.head()
# Drop Date_of_Journey 
train.drop('Date_of_Journey',axis = 1,inplace = True)
# lets convert the type of 'Dep_time' feature into datatime feature
train['Dep_Time'] = pd.to_datetime(train['Dep_Time'])
train['Departure_hr'] = train['Dep_Time'].dt.hour
train['Departure_min'] = train['Dep_Time'].dt.minute
# Drop dep_time
train.drop('Dep_Time',axis = 1,inplace = True)
train.head()
train['Arrival_hr'] = pd.to_datetime(train.Arrival_Time).dt.hour
train['Arrival_min'] = pd.to_datetime(train.Arrival_Time).dt.minute
# Drop Arrival_time
train.drop('Arrival_Time',axis = 1,inplace = True)
train.head()
train.Duration[:10]
duration = list(train.Duration)
for i in range(len(duration)):
    if len(duration[i].split(' ')) != 2:
        if 'h' in duration[i]:
            
            duration[i] = duration[i] + ' 0m'
        else:
            duration[i] = '0h ' + duration[i]
    
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(duration[i].split(sep = 'h')[0])              # Extracting hours
    duration_mins.append(duration[i].split(sep = 'm')[0].split()[-1])                           
# Creating two new feature
train['Duration_hr'] = duration_hours
train['Duration_mins'] = duration_mins
# Dropping duration feature
train.drop('Duration',1,inplace = True)
train.head()
train.Additional_Info.value_counts()
train.Airline.value_counts()
# Barplot---
plt.figure(figsize=(20,12))
sns.barplot(x = 'Airline', y = 'Price', data = train)
plt.xticks(rotation = '45')
# Dummy Encoding on Airline feature

Airline = train[['Airline']]
Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()
train.Source.value_counts()
# Barplot---
sns.barplot(x = 'Source', y = 'Price', data = train)
# Dummy Encoding
Source = train[['Source']]
Source = pd.get_dummies(Source, drop_first=True)
Source.head()
train.Destination.value_counts()
# Barplot
sns.barplot(x = 'Destination', y = 'Price', data = train)
# Dummy Encoding
Destination = train[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)
Destination.head()
# Route and total_stops are interrelated
# Additional_info has almost 80 percent data missing
# Dropping feartures
train.drop(['Route','Additional_Info'],axis = 1, inplace = True)
train.Total_Stops.value_counts()
# Barplot
sns.barplot(x = 'Total_Stops', y = 'Price', data = train)
train.head()
train.Total_Stops.value_counts()
train.replace({'non-stop': 0,'1 stop': 1,'2 stops': 2,'3 stops': 3,'4 stops': 4}, inplace = True)
train.Total_Stops.value_counts()
train.head()
# Concatenating all the dataframes together
data_train = pd.concat([train,Airline,Source,Destination],axis = 1)
data_train.head()
# Dropping Source , Airline and Destination as well
data_train.drop(['Airline','Source','Destination'], axis =1 , inplace = True)
data_train.head()
test = pd.read_excel('../input/flight-fare-prediction-mh/Test_set.xlsx')
test.head()
mn.matrix(test)
test.shape
test.dropna(inplace = True)
test['Journey_Date'] = pd.to_datetime(test.Date_of_Journey, format = '%d/%m/%Y').dt.day
test['Journey_Month'] = pd.to_datetime(test.Date_of_Journey, format = '%d/%m/%Y').dt.month
# Drop Date_of_Journey 
test.drop('Date_of_Journey',axis = 1,inplace = True)
test['Departure_hr'] = pd.to_datetime(test.Dep_Time).dt.hour
test['Departure_min'] = pd.to_datetime(test.Dep_Time).dt.minute
# Drop dep_time
test.drop('Dep_Time',axis = 1,inplace = True)
test['Arrival_hr'] = pd.to_datetime(test.Arrival_Time).dt.hour
test['Arrival_min'] = pd.to_datetime(test.Arrival_Time).dt.minute
# Drop Arrival_time
test.drop('Arrival_Time',axis = 1,inplace = True)
duration = list(test.Duration)

for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        else:
            duration[i] = '0h ' + duration[i]
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(duration[i].split(sep = 'h')[0])              # Extracting hours
    duration_mins.append(duration[i].split(sep = 'm')[0].split()[-1])   # Extracting mins
# Creating two new feature
test['Duration_hr'] = duration_hours
test['Duration_mins'] = duration_mins
test.drop('Duration',1,inplace = True)
# Creating dummy variables
Airline = test[['Airline']]
Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()

Source = test[['Source']]
Source = pd.get_dummies(Source, drop_first=True)
Source.head()

Destination = test[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)
Destination.head()

test.drop(['Route','Additional_Info'],axis = 1, inplace = True)

test.replace({'non-stop': 0,'1 stop': 1,'2 stops': 2,'3 stops': 3,'4 stops': 4}, inplace = True)

# Concatenating all the dataframes together

data_test = pd.concat([test,Airline,Source,Destination],axis = 1)

# Dropping Source , Airline and Destination as well
data_test.drop(['Airline','Source','Destination'], axis =1 , inplace = True)
data_test.head()
data_train.isnull().any()
data_train.columns
# Taking all the columns except dependent feature 'Price'
X = data_train.loc[:,['Total_Stops','Journey_Month', 'Departure_hr',
       'Departure_min', 'Arrival_hr', 'Arrival_min', 'Duration_hr',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()
y = data_train.pop('Price')
y = pd.DataFrame(y)
y.head()
y.shape
plt.figure(figsize=(20,18))
sns.heatmap(train.corr(),annot=True)
# Important featur using extra tree regressor
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,y)
#plot graph of feature importances for better visualization
plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
# Lets use RFE to get the top 15 features
lr_1 = LinearRegression()
lr_1.fit(X,y)
from sklearn.feature_selection import RFE 

rfe = RFE(lr_1,15)
rfe.fit(X,y)
# List of 15 columns selected by RFE
list(zip(X.columns,rfe.support_,rfe.ranking_))
col = X.columns[rfe.support_]
col
X_new_train =  X[col]
X_new_train.shape
lr1 = LinearRegression()
lr1.fit(X_new_train,y)
X_new_test = data_test[col]
data_test.head()
prediction = lr1.predict(X_new_test)
prediction.shape
prediction_train = lr1.predict(X_new_train)
from sklearn.metrics import r2_score
r2 = r2_score(y,prediction_train)
r2
lm = sm.add_constant(X_new_train)
lr2 = sm.OLS(y,lm).fit()
lr2.summary()
# Removing 'Destination_Delhi' because of high p value
X_new_train = X_new_train.drop(['Destination_Delhi'],axis = 1)
lm = sm.add_constant(X_new_train)
lr3 = sm.OLS(y,lm).fit()
lr3.summary()
# Calculate the VIFs for the new model
vif = pd.DataFrame()
X = X_new_train
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
