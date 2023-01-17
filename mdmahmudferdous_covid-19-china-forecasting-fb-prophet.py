import warnings

warnings.filterwarnings('ignore')
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
train=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
country=train[(train['Country/Region']=='Mainland China')]

country=country.groupby(country.ObservationDate).sum()

country['ObservationDate']=country.index
country.tail()
country_cc=country[['ObservationDate','Confirmed']]

country_cc['ds']=country_cc['ObservationDate']

country_cc['y']=country_cc['Confirmed']

country_cc.drop(columns=['ObservationDate','Confirmed'], inplace=True)

country_cc.head()



from fbprophet import Prophet

model_cc=Prophet()

model_cc.fit(country_cc)



future = model_cc.make_future_dataframe(periods=100)



forecast=model_cc.predict(future)



fig_Confirmed = model_cc.plot(forecast,xlabel = "Date",ylabel = "Confirmed")
country_cc=country[['ObservationDate','Deaths']]

country_cc['ds']=country_cc['ObservationDate']

country_cc['y']=country_cc['Deaths']

country_cc.drop(columns=['ObservationDate','Deaths'], inplace=True)

country_cc.head()



model_cc=Prophet()

model_cc.fit(country_cc)



future = model_cc.make_future_dataframe(periods=100)



forecast=model_cc.predict(future)



fig_Confirmed = model_cc.plot(forecast,xlabel = "Date",ylabel = "Deaths")