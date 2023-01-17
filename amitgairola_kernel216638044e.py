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



from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV

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
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")
train['Province_State'] = train['Province_State'].apply(lambda x: "CountryLevel" if pd.isna(x) else x)

test['Province_State'] = test['Province_State'].apply(lambda x: "CountryLevel" if pd.isna(x) else x)
train['Day'] = train['Date'].apply(lambda x: num_days('2020-01-22',x))

test['Day'] = test['Date'].apply(lambda x: num_days('2020-01-22',x))
unique_countries = train['Country_Region'].unique()
import math

def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    return np.sqrt(np.mean((np.log(1+y) - np.log(1+y_pred))**2))
train.info()
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

        

        ## Confirmed Cases

        polynomial_features= PolynomialFeatures(degree=5)

        x_train = tdf['Day'].values



        y_train = tdf['ConfirmedCases'].values

        x_train = x_train[:, np.newaxis]

        y_train = y_train[:, np.newaxis]



        x_poly = polynomial_features.fit_transform(x_train)

        x_poly1 = xgb.DMatrix(x_poly)

        param = {'max_depth':2, 'eta':1, 'objective':'linear' }

        #model = LinearRegression()

        #model = DecisionTreeRegressor()

        model = RandomForestRegressor()

        num_round = 2

        #model = xgb.XGBRFRegressor()

        clf = GridSearchCV(model,

                           {'max_depth': [2,4,6]

                            }, verbose=1)        

        clf.fit(x_poly, y_train)

        y_insample_pred = clf.predict(x_poly)

        print("RMSE : "+str(np.sqrt(mean_squared_error(y_train,y_insample_pred))))

        print("R-sq : "+str(np.sqrt(r2_score(y_train,y_insample_pred))))

        print("RMSLE : "+str(np.sqrt(rmsle(y_train,y_insample_pred))))



        x_test = testdf['Day']

        testdf['ConfirmedCases'] = clf.predict(polynomial_features.fit_transform(x_test[:,np.newaxis]))



        polynomial_features1= PolynomialFeatures(degree=4)

        y_train = tdf['Fatalities'].values

        y_train = y_train[:, np.newaxis]        

        #model = DecisionTreeRegressor()

        model = RandomForestRegressor()

        #model = xgb.XGBRFRegressor()

        clf = GridSearchCV(model,

                           {'max_depth': [2,4,6]

                            }, verbose=1)         

        clf.fit(x_poly, y_train)

        y_insample_pred = clf.predict(x_poly)

        print("RMSE : "+str(np.sqrt(mean_squared_error(y_train,y_insample_pred))))

        print("R-sq : "+str(np.sqrt(r2_score(y_train,y_insample_pred))))

        print("RMSLE : "+str(np.sqrt(rmsle(y_train,y_insample_pred))))

        x_test = testdf['Day']

        testdf['Fatalities'] = clf.predict(polynomial_features.fit_transform(x_test[:,np.newaxis]))

        predict_df = pd.concat([predict_df,testdf],axis=0)
predict_df.ConfirmedCases = predict_df.ConfirmedCases.apply(lambda x: np.round(x,0))

predict_df.Fatalities = predict_df.Fatalities.apply(lambda x: np.round(x,0))
predict_df[['ForecastId','ConfirmedCases','Fatalities']].to_csv("submission.csv",index=False)
newdf = predict_df.merge(right=train, how="inner",on=['Province_State', 'Country_Region', 'Date'])

newdf.drop(['Day_x','Day_y','Id','ForecastId'],axis=1, inplace=True)

newdf['ConfirmedCases_LSE'] = (np.log(newdf.ConfirmedCases_x+1)-np.log(newdf.ConfirmedCases_y+1))**2

newdf['Fatalities_LSE'] = (np.log(newdf.Fatalities_x+1)-np.log(newdf.Fatalities_y+1))**2

print(np.sqrt(np.sum(newdf.ConfirmedCases_LSE)/len(newdf)))

print(np.sqrt(np.sum(newdf.Fatalities_LSE)/len(newdf)))