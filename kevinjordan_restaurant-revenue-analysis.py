#loading necessary modules



import numpy as np 

import pandas as pd 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import math



#loading train and test csv file

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

#storing test Id for future submission 

testid = test.Id



#converting date into standard format 

train['Open Date'] = pd.to_datetime(train['Open Date'], format='%m/%d/%Y')

test['Open Date'] = pd.to_datetime(test['Open Date'], format='%m/%d/%Y')



#print(train['Open Date'].max())

#print(test['Open Date'].max())



#calculating lastdate and converting it into standard format



dateLastTrain = pd.DataFrame({'Date':np.repeat(['01/01/2017'],[len(train)]) })

dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y')

dateLastTest = pd.DataFrame({'Date':np.repeat(['01/01/2017'],[len(test)]) })

dateLastTest['Date'] = pd.to_datetime(dateLastTest['Date'], format='%m/%d/%Y')  



#calculating number of opendays

train['OpenDays'] = (dateLastTrain['Date'] - train['Open Date'])

test['OpenDays'] = (dateLastTest['Date'] - test['Open Date'])



#converting into integer

train['Days'] = train['OpenDays'] / np.timedelta64(1, 'D')

test['Days'] = test['OpenDays'] / np.timedelta64(1, 'D')



#creating dummies for City Group

citygroupDummy = pd.get_dummies(train['City Group'])

train = train.join(citygroupDummy)

citygroupDummy = pd.get_dummies(test['City Group'])

test = test.join(citygroupDummy)







#deleting unnecessary attributes

train.drop(['City Group','Open Date','City','Type','OpenDays','Id'],axis=1,inplace=True)

test.drop(['City Group','Open Date','City','Type','OpenDays','Id'],axis=1,inplace=True)



#seperating features and target in train file

target = train.revenue

train.drop('revenue',axis=1,inplace=True)





#train_x,test_x,train_y,test_y = train_test_split(train,target,random_state=40,test_size=0.3)

#print(train.shape,test.shape)



#Model selection fitting and calculating value for test file



#LinearRegression

from sklearn.linear_model import LinearRegression

model_log = LinearRegression()

model_log.fit(train,target)

print(model_log.score(train,target))

predicted = model_log.predict(test)

#print(len(predicted))



#RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=150)

model_RFR.fit(train,target)

model_RFR.score(train,target)



submission = pd.DataFrame({

        "Id": testid,

        "Prediction": predicted

    })



submission.to_csv('RestaurantRevenuePrediction.csv',header=True, index=False)