#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Wed Mar 25 09:37:58 2020



@author: juliogabrielchilela1

"""



"""

Complementary forecasting task to predict COVID-19 spread. 

This task is based on various regions across the world.

the COVID-19 Open Research Dataset (CORD-19) to attempt to address key 

open scientific questions on COVID-19. Those questions are drawn from 

National Academies of Sciences, Engineering, and Medicine’s (NASEM) 

and the World Health Organization (WHO).





While the challenge involves forecasting confirmed cases and fatalities 

between March 25 and April 22 by region, the primary goal isn't to produce 

accurate forecasts. It’s to identify factors that appear to impact the 

transmission rate of COVID-19.

"""

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



##############################################################################

# SECTION 1: IMPORTS

##############################################################################



import warnings #ignre warnings

warnings.filterwarnings("ignore")



import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



import os



#finalout = pd.merge(ds_test, dst, on='Country_Region', how='left')





#os.chdir("/Users/juliogabrielchilela1/Documents/Angola Cables/Covid19-AI/datasource-week-3") # Path to the Folder



#train = pd.read_csv('final_data/train.csv')

#test = pd.read_csv('final_data/test.csv')



#train.columns

#test.columns





##############################################################################

# READ EMPTY COUNTRIES

##############################################################################



#ler os paises e dados

df = pd.read_csv('/kaggle/input/trainsingle/trainSingle.csv')

df = pd.DataFrame(data=df) # Convert it to dataframe



#ler os treinos e o test

ds = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv') #.to_csv("tests.csv", index=False) #import the dataset

ds_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv') 



ds = pd.DataFrame(data=ds) # Convert it to dataframe

ds_test = pd.DataFrame(data=ds_test) # Convert it to dataframe



#Deletar conf cases e fatalities do dataframe geral 



del df['ConfirmedCases']

del df['Fatalities']







#Unir o valor do pais ao treino e ao test





ds = pd.merge(ds, df, on='Country_Region', how='left')

ds_test = pd.merge(ds_test, df, on='Country_Region', how='left')



del ds['Province_State_y']



del ds_test['Province_State_y']













ds_test.columns



ds = ds[['Province_State_x', 'Country_Region', 'Date', 'ConfirmedCases',

       'Fatalities', 'Perfomed_Test', 'Density', 'Population', 'GrowthRate',

       'Median', 'Median.Male', 'Median.Female', 'temp', 'min', 'max', 'stp',

       'GDP2018', 'Nurse_midwife_per_1000_2009.18', 'sex64', 'sex65plus',

       'Male.Lung', 'Total.Recovered']]





ds_test = ds_test[['ForecastId', 'Province_State_x', 'Country_Region', 'Date', 'Perfomed_Test', 'Density',

       'Population', 'GrowthRate', 'Median.Male', 'Median.Female', 'temp',

       'min', 'max', 'stp', 'GDP2018', 'Nurse_midwife_per_1000_2009.18',

       'sex64', 'sex65plus', 'Male.Lung', 'Total.Recovered']]







#ds.head()

#ds.columns









#ds.values



ds = ds.fillna(0)



#ds.values





#df = 



#ds.fillna(ds.mean(), inplace=True) # Fill the empty spaces with the mean of the column





X = ds[['Province_State_x', 'Country_Region', 'Date', 'Perfomed_Test', 'Density', 'Population', 'GrowthRate',

       'Median', 'Median.Male', 'Median.Female', 'temp', 'min', 'max', 'stp',

       'GDP2018', 'Nurse_midwife_per_1000_2009.18', 'sex64', 'sex65plus',

       'Male.Lung', 'Total.Recovered']]







##########

#Replace NAN with 0

#####

X = X.fillna(0)

#ENCODING THE COUNTRY

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

X["Country_Region"] = encoder.fit_transform(X["Country_Region"])





X['Province_State_x'] =  pd.to_numeric(X['Province_State_x'], errors='coerce')





X["Province_State_x"] = encoder.fit_transform(X["Province_State_x"])



#X['Province/State'] = pd.to_numeric(X['Province/State'], errors='coerce')





X['Date'] = pd.to_datetime(X.Date)



#X.columns

#del X['Id']













X["Day"] = X.Date.dt.day

X["Month"] = X.Date.dt.month

X["Year"] = X.Date.dt.year



del X['Date']







#NORMALIZING USING MINMAX

from sklearn.preprocessing import MinMaxScaler

X = MinMaxScaler().fit_transform(X)









y1 = ds[['ConfirmedCases']]

y2 = ds[['Fatalities']]







#print(ds.describe())

#print(ds.columns)

#.fillna(0)

#X.values











# define base model

def baseline_model():

    model = Sequential()

    model.add(Dense(2, input_dim=22, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

	# Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    #model.save('model_split5.h5')

    return model





# evaluate model

estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)

kfold = KFold(n_splits=5)

results = cross_val_score(estimator, X, y1, cv=kfold)

print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))





estimator.fit(X, y1)



#ds_test = pd.read_csv('final_data/test.csv') #.to_csv("tests.csv", index=False) #import the dataset

#ds_test2 = pd.read_csv('final_data/test.csv')

#del ds_test['Id']

#ENCODING THE COUNTRY

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

ds_test["Country_Region"] = encoder.fit_transform(ds_test["Country_Region"])





ds_test['Province_State_x'] =  pd.to_numeric(ds_test['Province_State_x'], errors='coerce')





ds_test["Province_State_x"] = encoder.fit_transform(ds_test["Province_State_x"])



#X['Province/State'] = pd.to_numeric(X['Province/State'], errors='coerce')





ds_test['Date'] = pd.to_datetime(ds_test.Date)



#print(ds_test.head(n=20))





ds_test["Day"] = ds_test.Date.dt.day

ds_test["Month"] = ds_test.Date.dt.month

ds_test["Year"] = ds_test.Date.dt.year





del ds_test['Date']

#del ds_test['Date_caso']

#del ds_test['ConfirmedCases']

#del ds_test['Fatalities']



#ds_test.columns

from sklearn.preprocessing import MinMaxScaler

ds_test = MinMaxScaler().fit_transform(ds_test)







prediction = estimator.predict(ds_test)

#accuracy_score(Y_test, prediction)



y_res = pd.DataFrame(prediction)



ds_test2 = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

ForecastId=pd.DataFrame(ds_test2, columns=["ForecastId"])



#GET THE FATALITIES ON SUBMISSION



#fat = pd.read_csv('submission.csv') #.to_csv("tests.csv", index=False) #import the dataset



ConfirmedCases=y_res

















#FATALITIES

# define base model

def baseline_model():

    model = Sequential()

    model.add(Dense(2, input_dim=22, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

	# Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    #model.save('model_split5.h5')

    return model





# evaluate model

estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)

kfold = KFold(n_splits=5)

results = cross_val_score(estimator, X, y2, cv=kfold)

print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))





estimator.fit(X, y2)





prediction = estimator.predict(ds_test)

#accuracy_score(Y_test, prediction)



y_res2 = pd.DataFrame(prediction)



#ds_test2 = pd.read_csv('test.csv')



#GET THE FATALITIES ON SUBMISSION





Fatalities=y_res2





junt=[ForecastId,ConfirmedCases, Fatalities]









sub= pd.concat(junt, axis=1)

sub.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities'] 

sub = sub.fillna(0)

sub.to_csv('/kaggle/working/submission.csv',index=False)






