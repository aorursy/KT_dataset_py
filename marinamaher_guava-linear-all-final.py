from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from matplotlib import pyplot

from sklearn.neighbors import KNeighborsRegressor

from datetime import datetime

from sklearn.tree import DecisionTreeRegressor

from numpy import absolute

from numpy import mean

from numpy import std

from sklearn.datasets import make_regression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedKFold

import pandas as pd

import os

import numpy as np 

import seaborn as sns 

import matplotlib.pyplot as plt 

from sklearn import preprocessing, svm 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn import metrics

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import IsolationForest

from sklearn.metrics import mean_absolute_error

from sklearn.covariance import EllipticEnvelope

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.svm import OneClassSVM

from sklearn.datasets import make_blobs

from numpy import quantile, where, random

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import cross_val_score

from sklearn.metrics import fbeta_score, make_scorer
df_in = pd.DataFrame()

df_out = pd.DataFrame()



for file in os.listdir(r"../input/guava-dataset"):

    if file.endswith('.xlsx'):

        excel_file = pd.ExcelFile("../input/guava-dataset/" + str(file))

        sheets = excel_file.sheet_names

        for sheet in sheets: # loop through sheets inside an Excel file

            if(sheet=="weather"):

                df = excel_file.parse(sheet_name = sheet)

                df_in = df_in.append(df)  

            

            if(sheet=="population"):

                df = excel_file.parse(sheet_name = sheet)

                df_out = df_out.append(df)  

col_names = df_in.columns

col_names=list(col_names)



s="Unnamed"

for i in col_names :

    if(str(i).find(s)==0):

        if( (col_names[col_names.index(i)+1]).find(s)==0 ):

            name=col_names[col_names.index(i)-1]

            col_names[col_names.index(i)]   = name +"_avg"

            col_names[col_names.index(name +"_avg")+1] = name +"_low"

            col_names[col_names.index(name +"_avg")-1] = name +"_high"

        elif( col_names[col_names.index(i)-1]== "Wind (km/h)" ):

            col_names[col_names.index(i)-1]= "Wind (km/h)"+"_high"

            col_names[col_names.index("Wind (km/h)"+"_high")+1]= "Wind (km/h)"+"_avg"



col_names2 = df_in.columns

col_names2 = list(col_names2)



d=dict(zip(col_names2,col_names))

df_in.rename(columns = d, inplace = True)

df_in=df_in.drop(0,axis=0)

df_in=df_in.drop(['Temp. (°C)_high','Temp. (°C)_low','Dew Point\xa0(°C)_low','Dew Point\xa0(°C)_high','Humidity\xa0(%)_high',

                    'Humidity\xa0(%)_low','Sea Level Pressure (hPa)_high','Sea Level Pressure (hPa)_low','Visibility\xa0(km)_high',

                     'Visibility\xa0(km)_low','Wind (km/h)_high','Gust Speed (km/h)','Events','Precip. (mm)'], axis=1)



df_in.head(2)
mapping = {'rain': 1, 'fog': 2 , 'Rain': 1, 'Fog': 2 , '\xa0 ' : 0}

df_in=df_in.replace({'Events': mapping})

 

df_in=df_in.fillna(0)

print("NA :" , df_in.isna().sum().sum())
df_out=df_out.drop(0,axis=0)

out_names = df_out.iloc[0,:]

out_names=out_names.tolist()

out_names[0]="Date"



col_names_out = df_out.columns

col_names_out = list(col_names_out)



d=dict(zip(col_names_out,out_names))

df_out.rename(columns = d, inplace = True)

df_out=df_out.drop(1,axis=0)

df_out.head(2)
nan_array = np.empty((df_in.shape[1]))

nan_array[:] = np.NaN



dfs = list()

j=0

step=55



for i in range(df_in.shape[0]):

    if( i==7 or i==step):

        dfs.append( df_out.iloc[j,:] )  

        j=j+1

        step=i+14

        

    else:

        dfs.append( nan_array )  

    



df_out_new = pd.DataFrame(dfs)
df_out_new.drop(0, axis='columns', inplace=True)

df_in.drop(2014, axis='columns', inplace=True)

l=list(df_in.columns)

print("in:",df_in.shape ,"   ","out:", df_out_new.shape)



df_out_new=df_out_new.interpolate(method='nearest')

df_out_new=df_out_new.fillna(0)
errors = list()

for i in range(df_out.shape[1]-1):

    X=df_in

    y=df_out_new.iloc[:,i]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) 



    ee=EllipticEnvelope(contamination=0.2,random_state=42)

    yhat = ee.fit_predict(X_train)

    mask = yhat != -1



    X_train, y_train = X_train[mask], y_train[mask]

    r1 = KNeighborsRegressor()

    r2 = ExtraTreesRegressor(n_estimators=9, random_state=1)

    er = VotingRegressor([('lr', r1), ('ef', r2)])



    er.fit(X_train, y_train)

    y_pred = er.predict(X_test)

    error=np.sqrt(metrics.mean_squared_error(y_test, y_pred))*100/np.mean(y_test)

    errors.append(error)



print('Root Mean Squared Error:', mean(errors))

    
model = DecisionTreeRegressor()

model.fit(X, y)

importance = model.feature_importances_

for i,v in enumerate(importance):

    print('Feature: %0d, Score: %.5f' % (i,v),"    ",l[i])

#pyplot.bar([x for x in range(len(importance))], importance)

#pyplot.show()