import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

train=pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')

test_data=pd.read_excel('../input/flight-fare-prediction-mh/Test_set.xlsx')
pd.set_option('display.max_columns',None)
train.tail()
(10862*100)/10863
train.info()



#Luckily we have a 99.99 % of non null values so we can drop the null value.
train.dropna(inplace=True)
train.info()
train['Journey_day']=pd.to_datetime(train.Date_of_Journey,format='%d/%m/%Y').dt.day



train['Journey_month']=pd.to_datetime(train.Date_of_Journey,format='%d/%m/%Y').dt.month



train.drop(train[['Date_of_Journey']],axis=1,inplace=True)
train['DEP_Hr']=pd.to_datetime(train.Dep_Time).dt.hour



train['DEP_min']=pd.to_datetime(train.Dep_Time).dt.minute



train.drop(train[['Dep_Time']],axis=1,inplace=True)
train['Arrival_Hour']=pd.to_datetime(train.Arrival_Time).dt.hour

train['Arrival_Min']=pd.to_datetime(train.Arrival_Time).dt.minute

train.drop(train[['Arrival_Time']],axis=1,inplace=True)
train['Additional_Info'].value_counts()
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

    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))
train['Duration_hours']=duration_hours

train['Duration_mins']=duration_mins
train.drop(train[['Duration']],axis=1,inplace=True)
# For which Destination is priced high

plt.figure(figsize=(10,10))

sns.catplot(x='Destination',y='Price',data=train,kind='boxen',height=5,aspect=3)

plt.title("Destination Vs Price",loc='center',fontsize=20)

plt.show()
#From which source is the price high

sns.catplot(x='Source',y='Price',data=train,aspect=3,kind='point',height=5)

plt.title("Source Vs Price",loc='center',fontsize=20)

plt.show()
#Which airways has highest rate

sns.catplot(x='Airline',y='Price',kind='point',data=train,height=5,aspect=3)

plt.xticks(rotation=45)

plt.show()
train.drop(train[['Additional_Info','Route']],axis=1,inplace=True)
#Now comes encoding part

#train.info()

cat_cols=train.select_dtypes(include=[np.object])

print(cat_cols.columns)
#df_train=pd.get_dummies(train,drop_first=True)
train.head()
train.columns
train.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
Source = train[["Source"]]



Source = pd.get_dummies(Source, drop_first= True)



Source.head()
Destination = train[["Destination"]]



Destination = pd.get_dummies(Destination, drop_first = True)



Destination.head()
Airline = train[["Airline"]]



Airline = pd.get_dummies(Airline, drop_first= True)



Airline.head()
final_train = pd.concat([train, Airline, Source, Destination], axis = 1)
final_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
final_train
test_data['Journey_day']=pd.to_datetime(test_data.Date_of_Journey,format='%d/%m/%Y').dt.day



test_data['Journey_month']=pd.to_datetime(test_data.Date_of_Journey,format='%d/%m/%Y').dt.month



test_data.drop(test_data[['Date_of_Journey']],axis=1,inplace=True)
test_data['DEP_Hr']=pd.to_datetime(test_data.Dep_Time).dt.hour



test_data['DEP_min']=pd.to_datetime(test_data.Dep_Time).dt.minute



test_data.drop(test_data[['Dep_Time']],axis=1,inplace=True)
test_data['Arrival_Hour']=pd.to_datetime(test_data.Arrival_Time).dt.hour

test_data['Arrival_Min']=pd.to_datetime(test_data.Arrival_Time).dt.minute

test_data.drop(test_data[['Arrival_Time']],axis=1,inplace=True)
test_data['Additional_Info'].value_counts()
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

    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))
test_data['Duration_hours']=duration_hours

test_data['Duration_mins']=duration_mins
test_data.drop(test_data[['Duration']],axis=1,inplace=True)
test_data.drop(test_data[['Additional_Info','Route']],axis=1,inplace=True)
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

Source = pd.get_dummies(test_data["Source"], drop_first= True)

Airline = pd.get_dummies(test_data["Airline"], drop_first= True)
final_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)
final_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
final_test
final_train.drop(final_train[['Airline_Trujet']],axis=1,inplace=True)
len(final_train.columns)
final_test.columns=['Total_Stops', 'Journey_day', 'Journey_month', 'DEP_Hr',

       'DEP_min', 'Arrival_Hour', 'Arrival_Min', 'Duration_hours',

       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',

       'Airline_Jet Airways', 'Airline_Jet Airways Business',

       'Airline_Multiple carriers',

       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',

       'Airline_Vistara', 'Airline_Vistara Premium economy',

       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',

       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',

       'Destination_Kolkata', 'Destination_New Delhi']
final_train
final_test
X=final_train.drop(train[['Price']],axis=1)
Y=final_train['Price']
#To validate we will split the train data into train and test splits

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
print('X_train shape: ',X_train.shape)

print('Y_train shape: ',Y_train.shape)

print('X_test shape: ',X_test.shape)

print('Y_test shape: ',Y_test.shape)
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor 
from xgboost import XGBRegressor
rf=RandomForestRegressor()

gb=GradientBoostingRegressor()

xgb=XGBRegressor()
rf.fit(X_train,Y_train)
gb.fit(X_train,Y_train)
xgb.fit(X_train,Y_train)
pred=rf.predict(X_test)
pred_gb=gb.predict(X_test)
pred_XGB=xgb.predict(X_test)
from sklearn.metrics import accuracy_score
print('------------------------------------------RF Scores---------------------------------------------')

print('Train_SCore :',rf.score(X_train,Y_train))

print('Test_SCore :' ,rf.score(X_test,Y_test))

print('------------------------------------------GB Scores---------------------------------------------')

print('Train_SCore : ',gb.score(X_train,Y_train))

print('Test_SCore : ' ,gb.score(X_test,Y_test))

print('------------------------------------------XGB Scores---------------------------------------------')

print('Train_SCore :',xgb.score(X_train,Y_train))

print('Test_SCore :' ,xgb.score(X_test,Y_test))
plt.figure(figsize=(10,7))

sns.distplot(Y_test-pred)

plt.title('Random Forest')
plt.figure(figsize=(10,7))

sns.distplot(Y_test-pred_gb)

plt.title('Gradient Boosting')
plt.figure(figsize=(10,7))

sns.distplot(Y_test-pred_XGB)

plt.title('XGBoosting')
plt.scatter(Y_test,pred)
import plotly.express as px
fig=px.scatter(x=Y_test,y=pred)



fig.show()
fig=px.scatter(x=Y_test,y=pred_gb)



fig.show()
fig=px.scatter(x=Y_test,y=pred_XGB)

fig.show()
from sklearn.model_selection import RandomizedSearchCV
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]

max_features=['auto','sqrt']

max_depth=[int(x) for x in np.linspace(5,30,num=7)]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]
rf_grid={'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}
rf_random=RandomizedSearchCV(estimator=rf,param_distributions=rf_grid,scoring='neg_mean_squared_error',n_iter=10,

                             cv=5,verbose=2,random_state=42,n_jobs=1)
rf_random.fit(X_train,Y_train)
xgb_random=RandomizedSearchCV(estimator = xgb, param_distributions = rf_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2,

                                random_state=42, n_jobs = 1)
xgb_random.fit(X_train,Y_train)
rf_random.best_params_
predRF=rf_random.predict(X_test)
plt.figure(figsize = (8,8))

sns.distplot(Y_test-predRF)

plt.show()
xgb_random.best_params_
predXGB=xgb_random.predict(X_test)
plt.figure(figsize = (8,8))

sns.distplot(Y_test-predXGB)

plt.show()
rf_random.predict(final_test)
xgb_random.predict(final_test)