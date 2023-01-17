# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')



# Replace missing data by country

train['Province_State'] = train['Province_State'].replace(np.nan, 'nan', regex=True)

train['Province_State'] = np.where(train['Province_State'] == 'nan' ,train['Country_Region'],train['Province_State'])



test['Province_State'] = test['Province_State'].replace(np.nan, 'nan', regex=True)

test['Province_State'] = np.where(test['Province_State'] == 'nan' ,test['Country_Region'],test['Province_State'])



# Calculate Days by substract dates: current(date) - min(date)

train['Date'] = pd.to_datetime(train['Date']) # Transfer date to what Python can load

train['Days'] = train['Date']-train['Date'].min()

train['Days'] = train['Days'].dt.days.astype(int) #.astype(int) ->integer



# ConfirmedCase, Fatalities float->int

train[['ConfirmedCases', 'Fatalities']] = train[['ConfirmedCases', 'Fatalities']].astype(int)



test['Date'] = pd.to_datetime(test['Date'])

test['Days'] = test['Date']-train['Date'].min()

test['Days'] = test['Days'].dt.days.astype(int)



# For train set, every country and state has 84 days;

# Totally 313 Countries and States

# For test, every state and country has 43 days;

# Totoally 313 Countries and States



# Create a metrix countain 313 empty [0]

province_train = np.zeros(int(len(train)/84)).astype(object)

province_test = np.zeros(int(len(test)/43)).astype(object)



X_train = np.zeros(len(province_train)).astype(object)

y_train = np.zeros(len(province_train)).astype(object)

X_test = np.zeros(len(province_test)).astype(object)





province_index = 0 

index = 0

while province_index < len(train):

     province = train['Province_State'][province_index]

     country = train['Country_Region'][province_index]

     province_train[index] = train[(train['Province_State'] == province) & (train['Country_Region']== country)]

     X_train[index] = pd.DataFrame(province_train[index]).iloc[:,[0,1,2,6]]

     y_train[index] = pd.DataFrame(province_train[index]).iloc[:,4:6]

     index +=1

     province_index += 84     

     

province_index = 0 

index = 0

while province_index < len(test):

     province = test['Province_State'][province_index]

     country = test['Country_Region'][province_index]

     province_test[index] = test[(test['Province_State'] == province) & (test['Country_Region']== country)]

     X_test[index] = pd.DataFrame(province_test[index]).iloc[:,[0,1,2,4]]

     index +=1

     province_index += 43



# Labelencoder categorical values -> numerical values

from sklearn.preprocessing import LabelEncoder

label_encoder1 = LabelEncoder()

label_encoder2 = LabelEncoder()



# =============================================================================

# from xgboost import XGBRegressor

# =============================================================================

from sklearn.linear_model import LinearRegression # Linear Regression

from sklearn.preprocessing import PolynomialFeatures # Polynomial Regression



for index in range(0,len(province_train)):

     X_train[index]['Province_State'] = label_encoder1.fit_transform(X_train[index]['Province_State'])

     X_train[index]['Country_Region'] = label_encoder2.fit_transform(X_train[index]['Country_Region'])

     X_test[index]['Province_State'] = label_encoder1.fit_transform(X_test[index]['Province_State'])

     X_test[index]['Country_Region'] = label_encoder2.fit_transform(X_test[index]['Country_Region'])

     

# =============================================================================

#      # y_pred['ConfirmedCases']

#      model = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=15)

#      model.fit(X_train[index].iloc[:,1:4], y_train[index].iloc[:,0])

#      y_pred = model.predict(X_test[index].iloc[:,1:4])

#      y_pred = np.around(y_pred,decimals=0).astype(int)

#      X_test[index]['ConfirmedCases'] = np.where(y_pred <0 ,0,y_pred)

# 

#      # y_pred['Fatalities']

#      model = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=15)

#      model.fit(X_train[index].iloc[:,1:4], y_train[index].iloc[:,1])

#      y_pred = model.predict(X_test[index].iloc[:,1:4])

#      y_pred = np.around(y_pred,decimals=0).astype(int)    

#      X_test[index]['Fatalities'] = np.where(y_pred <0 ,0,y_pred)

# 

# =============================================================================

     model = PolynomialFeatures(degree = 4)

     X_poly = model.fit_transform(X_train[index].iloc[:,1:4])

     model.fit(X_poly, y_train[index].iloc[:,0])

     

     # Apply poly regression into linear regression

     lin_reg = LinearRegression()

     lin_reg.fit(X_poly, y_train[index].iloc[:,0])

     y_pred=lin_reg.predict(model.fit_transform(X_test[index].iloc[:,1:4]))

     y_pred = np.around(y_pred,decimals=0).astype(int)

     # Convert all negative values to zero

     X_test[index]['ConfirmedCases'] = np.where(y_pred < 0 ,0,y_pred) 



     # Predict fatalities

     model = PolynomialFeatures(degree = 4)

     X_poly = model.fit_transform(X_train[index].iloc[:,1:4])

     model.fit(X_poly, y_train[index].iloc[:,0])

     lin_reg = LinearRegression()

     lin_reg.fit(X_poly, y_train[index].iloc[:,1])

     y_pred=lin_reg.predict(model.fit_transform(X_test[index].iloc[:,1:4]))

     y_pred = np.around(y_pred,decimals=0).astype(int)

     X_test[index]['Fatalities'] = np.where(y_pred <0 ,0,y_pred)



     X_test[index]['Province_State'] = province_test[index].iloc[:,1]

     X_test[index]['Country_Region'] = province_test[index].iloc[:,2]

     

# Extract X_test

test_final = pd.DataFrame([])

for i in range(0, len(X_test)):

     # concatenate

    test_final = pd.concat([test_final, X_test[i]], ignore_index=True)



test['ConfirmedCases'] = test_final['ConfirmedCases']

test['Fatalities'] = test_final['Fatalities']



# Replace duplicated pred values by actually values

test[(test['Days'] >=71)&(test['Days']<=83)]['ConfirmedCases'] = train[(train['Days'] >=71)&(train['Days']<=83)]['ConfirmedCases']

test[(test['Days'] >=71)&(test['Days']<=83)]['Fatalities'] = train[(train['Days'] >=71)&(train['Days']<=83)]['Fatalities']



replace_train_index = train.index[(train['Days'] >=71)&(train['Days']<=83)].values

replace_test_index = test.index[(test['Days'] >=71)&(test['Days']<=83)].values



index = 0

for i in replace_test_index:

     test.set_value(i,'ConfirmedCases', train.iloc[replace_train_index[index],4])

     test.set_value(i,'Fatalities', train.iloc[replace_train_index[index],5])

     index += 1





submission = test.iloc[:,[0,5,6]]

submission.to_csv('submission.csv', index=False)



# =============================================================================

# vic = test[(test['Province_State'] == "Victoria") & (test['Country_Region']== "Australia")]

# plt.plot(vic["Date"], vic["ConfirmedCases"])

# 

# New_York = test[(test['Province_State'] == "New York") & (test['Country_Region']== "US")]

# plt.plot(New_York["Date"], New_York["ConfirmedCases"])

# 

# Italy = test[(test['Province_State'] == "Italy") & (test['Country_Region']== "Italy")]

# plt.plot(Italy["Date"], Italy["ConfirmedCases"])

# =============================================================================
