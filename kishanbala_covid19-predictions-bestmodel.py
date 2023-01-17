import numpy as np
import pandas as pd
train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv') #read csv with the column containing dates as datetime datatype
test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
sub_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
#Get the data types of all the columns in both the sets
display(train_data.dtypes)
display(test_data.dtypes)
#Check if NA's present in the data
display(train_data.isna().sum())
display(test_data.isna().sum())
#Since NAs are present in both train and test data in the column Province/State
#we drop that column
train_data = train_data.drop(['Province/State'], axis=1)
test_data = test_data.drop(['Province/State'], axis=1)
#Check the data types of all attributes
display(train_data.dtypes)
display(test_data.dtypes)
train_atrributes = list(train_data.columns)
test_attr = list(test_data.columns)
display(train_atrributes)
display(test_attr)
train_data = train_data[:-12] #Since the Confirmed cases and Fatalities need to be predicted
                          #from 12-03-2020 we drop those rows.
train_data['Date'] = train_data['Date'].apply(lambda x: x.replace("-",""))
train_data['Date'] = train_data['Date'].astype(int) 
test_data['Date'] = test_data['Date'].apply(lambda x: x.replace("-",""))
test_data['Date'] = test_data['Date'].astype(int)
X_train = train_data[['Lat', 'Long', 'Date' ]]
Y1 = train_data[['ConfirmedCases']]
Y2 = train_data[['Fatalities']]
X_test = test_data[['Lat', 'Long', 'Date']]
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

model = DecisionTreeRegressor(random_state=1, max_depth=15,splitter='best')
model.fit(X_train, Y1.values.ravel())
model.score(X_train, Y1)
model1 = DecisionTreeRegressor(max_depth=14,splitter='best')
model1.fit(X_train, Y2.values.ravel())
model1.score(X_train, Y2)
pred1 = model.predict(X_test)
pred2 = model1.predict(X_test)
pred1 = pd.DataFrame(pred1)
pred1.columns = ["ConfirmedCases_Prediction"]
pred1['ConfirmedCases_Prediction'] = pred1['ConfirmedCases_Prediction'].astype(int)
pred2 = pd.DataFrame(pred2)
pred2.columns = ["Fatality_Prediction"]
pred2['Fatality_Prediction'] = pred2['Fatality_Prediction'].astype(int)
sub_data.columns
sub_new = sub_data[["ForecastId"]]
OP = pd.concat([pred1,pred2,sub_new],axis=1)
OP.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']
OP = OP[['ForecastId','ConfirmedCases', 'Fatalities']]
OP.to_csv("submission.csv",index=False)