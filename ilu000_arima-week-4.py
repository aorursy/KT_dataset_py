print("Read in libraries")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA

from random import random
print("read in train file")

df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv",

               usecols=['Province_State','Country_Region','Date','ConfirmedCases','Fatalities'])
print("fill blanks and add region for counting")

df.fillna(' ',inplace=True)

df['Lat']=df['Province_State']+df['Country_Region']

df.drop('Province_State',axis=1,inplace=True)

df.drop('Country_Region',axis=1,inplace=True)
countries_list=df.Lat.unique()

df1=[]

for i in countries_list:

    df1.append(df[df['Lat']==i])

print("we have "+ str(len(df1))+" regions in our dataset")



#read in test file 

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
#create the estimates assuming measurement error 

submit_confirmed=[]

submit_fatal=[]

for i in df1:

    # contrived dataset

    data = i.ConfirmedCases.astype('int32').tolist()

    # fit model

    try:

        #model = SARIMAX(data, order=(2,1,0), seasonal_order=(1,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        model = SARIMAX(data, order=(1,1,0), seasonal_order=(1,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        #model = SARIMAX(data, order=(1,1,0), seasonal_order=(0,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        #model = ARIMA(data, order=(3,1,2))

        model_fit = model.fit(disp=False)

        # make prediction

        predicted = model_fit.predict(len(data), len(data)+34)

        new=np.concatenate((np.array(data),np.array([int(num) for num in predicted])),axis=0)

        submit_confirmed.extend(list(new[-43:]))

    except:

        submit_confirmed.extend(list(data[-10:-1]))

        for j in range(34):

            submit_confirmed.append(data[-1]*2)

    

    # contrived dataset

    data = i.Fatalities.astype('int32').tolist()

    # fit model

    try:

        #model = SARIMAX(data, order=(1,0,0), seasonal_order=(0,1,1,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        model = SARIMAX(data, order=(1,1,0), seasonal_order=(1,1,0,12),measurement_error=True)#seasonal_order=(1, 1, 1, 1))

        #model = ARIMA(data, order=(3,1,2))

        model_fit = model.fit(disp=False)

        # make prediction

        predicted = model_fit.predict(len(data), len(data)+34)

        new=np.concatenate((np.array(data),np.array([int(num) for num in predicted])),axis=0)

        submit_fatal.extend(list(new[-43:]))

    except:

        submit_fatal.extend(list(data[-10:-1]))

        for j in range(34):

            submit_fatal.append(data[-1]*2)
#make the submission file 

df_submit=pd.concat([pd.Series(np.arange(1,1+len(submit_confirmed))),pd.Series(submit_confirmed),pd.Series(submit_fatal)],axis=1)

df_submit=df_submit.fillna(method='pad').astype(int)
#view submission file 

df_submit.head()
#examine the test file 

test.head()
#join the submission file info to the test data set 

#rename the columns 

df_submit.rename(columns={0: 'ForecastId', 1: 'ConfirmedCases',2: 'Fatalities',}, inplace=True)



#join the two data items 

complete_test= pd.merge(test, df_submit, how="left", on="ForecastId")
#df_submit.interpolate(method='pad', xis=0, inplace=True)

df_submit.to_csv('submission.csv',header=['ForecastId','ConfirmedCases','Fatalities'],index=False)

complete_test.to_csv('complete_test.csv',index=False)