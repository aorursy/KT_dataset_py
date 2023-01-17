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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from datetime import date



from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV,BayesianRidge

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold, train_test_split, GridSearchCV



import xgboost as xgb



import sys



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")

import torch
def num_days(date1, date2):

    dt1 = date1.split("-")

    dt2 = date2.split("-")

    

    d2 = date(int(dt2[0]),int(dt2[1]),int(dt2[2]))

    d1 = date(int(dt1[0]),int(dt1[1]),int(dt1[2]))

              

    delta = d2-d1

    return(delta.days)
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
train['Province_State'] = train['Province_State'].apply(lambda x: "CountryLevel" if pd.isna(x) else x)

test['Province_State'] = test['Province_State'].apply(lambda x: "CountryLevel" if pd.isna(x) else x)
train['Day'] = train['Date'].apply(lambda x: num_days('2020-01-22',x))

test['Day'] = test['Date'].apply(lambda x: num_days('2020-01-22',x))
import math

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    return np.sqrt(np.mean((np.log(1+y) - np.log(1+y_pred))**2))
unique_countries = train['Country_Region'].unique()
predict_df = pd.DataFrame()

for country in unique_countries:

    localdf = train.copy()

    testdf = test.copy()

    localdf = localdf[localdf.Country_Region == country]

    unique_province = localdf['Province_State'].unique()

    for province in unique_province:

        print(country+" - "+province)

        tdf = localdf[localdf['Province_State']==province]

        testdf = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

        x_train = tdf['Day'].values

        max_len = tdf.shape[0]

        

        sample_train = tdf[tdf.Day <= max_len-10]

        

        sample_test = tdf[tdf.Day>max_len-10]

        sample_train_x = sample_train['Day'].values

        sample_train_y = sample_train['ConfirmedCases'].values

        sample_test_x = sample_test['Day'].values

        sample_test_y = sample_test['ConfirmedCases'].values        

        sample_train_x = sample_train_x[:, np.newaxis]

        sample_train_y = sample_train_y[:, np.newaxis]

        sample_test_x = sample_test_x[:, np.newaxis]

        sample_test_y = sample_test_y[:, np.newaxis]    

        #print("Train "+str(sample_train_x.shape))



        y_train = tdf['ConfirmedCases'].values

        x_train = x_train[:, np.newaxis]

        y_train = y_train[:, np.newaxis]



        ## Confirmed Cases

        rmse = 10000000

        for i in range(1,10):

            #print(i)

            polynomial_features= PolynomialFeatures(degree=i)

            x_poly = polynomial_features.fit_transform(sample_train_x)

            model = BayesianRidge()

            model.fit(x_poly, sample_train_y)

            y_insample_pred = model.predict(polynomial_features.transform(sample_test_x))

            l_rmse = np.sqrt(mean_squared_error(sample_test_y,y_insample_pred))

            print("i = "+str(i)+ " In Sample RMSE"+str(l_rmse)+" Old RMSE"+str(rmse))

            if l_rmse < rmse:

                #print("Assigning")

                rmse = l_rmse

                f_degree = i

        print("Degree of curve for confirmed cases = "+str(f_degree))

        polynomial_features = PolynomialFeatures(degree=f_degree)

        x_poly = polynomial_features.fit_transform(x_train)

        model = BayesianRidge()

        model.fit(x_poly, y_train)        

        x_test = testdf['Day']

        testdf['ConfirmedCases'] = model.predict(polynomial_features.fit_transform(x_test[:,np.newaxis]))

        

        ## Fatalities

        sample_train_y = sample_train['Fatalities'].values

        sample_test_y = sample_test['Fatalities'].values        

        sample_train_y = sample_train_y[:,np.newaxis]

        sample_test_y = sample_test_y[:,np.newaxis]



        y_train = tdf['Fatalities'].values

        y_train = y_train[:, np.newaxis]        

        rmse = 100000000

        for j in range(1,10):

            #print(j)

            polynomial_features= PolynomialFeatures(degree=j)

            x_poly = polynomial_features.fit_transform(sample_train_x)

            model = BayesianRidge()

            model.fit(x_poly, sample_train_y)

            y_insample_pred = model.predict(polynomial_features.transform(sample_test_x))

            l_rmse = np.sqrt(mean_squared_error(sample_test_y,y_insample_pred))

            print("J = "+str(j)+ " In Sample RMSE"+str(l_rmse)+" Old RMSE"+str(rmse))

            if l_rmse < rmse:

                #print("Assigning")

                rmse = l_rmse

                f_degree = j

        print("Degree of curve for fatalities cases = "+str(f_degree))

        polynomial_features1 = PolynomialFeatures(degree=f_degree)

        x_poly = polynomial_features1.fit_transform(x_train)

        model = BayesianRidge()

        model.fit(x_poly,y_train)

        x_test = testdf['Day']

        testdf['Fatalities'] = model.predict(polynomial_features1.fit_transform(x_test[:,np.newaxis]))

        predict_df = pd.concat([predict_df,testdf],axis=0)
predict_df.ConfirmedCases = predict_df.ConfirmedCases.apply(lambda x: np.round(x,0))

predict_df.Fatalities = predict_df.Fatalities.apply(lambda x: np.round(x,0))
predict_df[['ForecastId','ConfirmedCases','Fatalities']].to_csv("submission.csv",index=False)