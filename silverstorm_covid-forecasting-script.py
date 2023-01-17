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
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

df_train.head()
df_test.head()
train = df_train

test = df_test



# Converting to datetime64 and splitting off redundant year

train.Date = pd.to_datetime(train.Date)

test.Date = pd.to_datetime(test.Date)

train['Date'] = train['Date'].dt.strftime("%d%m").astype(int)

test['Date'] = test['Date'].dt.strftime("%d%m").astype(int)
# Filling NaNs with 'Data_NA' for Not Applicable

train.Province_State.fillna('Data_NA',inplace=True)

test.Province_State.fillna('Data_NA',inplace=True)
# Storing various countries in the Countries list

Countries = train.Country_Region.unique()

Countries
# Storing the Provinces of each country in a pandas series

Provinces = train.groupby('Country_Region')['Province_State'].unique()

Provinces
# To find number of predictors

len(Countries)*len(Provinces)
from xgboost import XGBRegressor



class COVID_19:

    def __init__(self):

        self.count = 0

        self.Predictions = {'ForecastId':[],'ConfirmedCases':[],'Fatalities':[]}

        

    def Forecast_COVID_19(self,N=324):

        for country in Countries:

            for province in Provinces[country]:

                country_df = train[(train.Country_Region==country)]

                province_df = country_df[(country_df.Province_State==province)]



                X_train = province_df.Date.values.reshape(-1,1)

                y_trainC = province_df.ConfirmedCases.values

                y_trainF = province_df.Fatalities.values



                test_country_df = test[(test.Country_Region==country)]

                test_province_df = test_country_df[(test_country_df.Province_State==province)]

                X_test = test_province_df.Date.values.reshape(-1,1)



                y_predC = XGBRegressor(n_estimators=N).fit(X_train, y_trainC).predict(X_test)

                y_predF = XGBRegressor(n_estimators=N).fit(X_train, y_trainF).predict(X_test)



                FC_ID = np.array(test_province_df.ForecastId)



                for i,(j,k) in enumerate(zip(y_predC,y_predF)):

                    self.Predictions['ForecastId'].append(FC_ID[i])

                    self.Predictions['ConfirmedCases'].append(np.round(j))

                    self.Predictions['Fatalities'].append(np.round(k))



        self.df_pred = pd.DataFrame(self.Predictions)



        if self.count==0:

            self.df_pred.to_csv('submission.csv',index=False)

        if self.count>=1:

            self.df_pred.to_csv('submission_'+str(self.count)+'.csv',index=False)



        self.count+=1
COVID_Forecaster = COVID_19()

COVID_Forecaster.Forecast_COVID_19(1000)