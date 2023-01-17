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


confirmed=pd.read_csv('/kaggle/input/covid-19/time_series_covid19_confirmed_global.csv')

deaths=pd.read_csv('/kaggle/input/covid-19/time_series_covid19_deaths_global.csv')

recovered=pd.read_csv('/kaggle/input/covid-19/time_series_covid19_recovered_global.csv')





# # Melt



# In[129]:





confirmed_1=pd.melt(confirmed, id_vars=['Province/State','Country/Region','Lat','Long'])

deaths_1=pd.melt(deaths, id_vars=['Province/State','Country/Region','Lat','Long'])

recovered_1=pd.melt(recovered, id_vars=['Province/State','Country/Region','Lat','Long'])

confirmed_1.rename(columns={'variable':'Date'}, inplace=True)

deaths_1.rename(columns={'variable':'Date'}, inplace=True)

recovered_1.rename(columns={'variable':'Date'}, inplace=True)

confirmed_2=confirmed_1

deaths_2=deaths_1

recovered_2=recovered_1





# # Join



# In[130]:





df=pd.DataFrame()

df['Province/State']=confirmed_1['Province/State']

df['Country/Region']=confirmed_1['Country/Region']

df['Lat']=confirmed_1['Lat']

df['Long']=confirmed_1['Long']

df['Date']=confirmed_1['Date']



df=pd.merge(df, confirmed_1[['Province/State','Country/Region','Date','value']], on=['Date','Country/Region','Province/State'] )

df.rename(columns={'value':'Confirmed'}, inplace=True)

df=pd.merge(df, deaths_1[['Province/State','Country/Region','Date','value']], on=['Date','Country/Region','Province/State'] )

df.rename(columns={'value':'Deaths'}, inplace=True)

df=pd.merge(df, recovered_1[['Province/State','Country/Region','Date','value']], on=['Date','Country/Region','Province/State'] )

df.rename(columns={'value':'Recovered'}, inplace=True)





# # Mortality



# In[131]:





df['Mortality_Rate(D/C)']=df['Deaths']/df['Confirmed']





# # Var.



# In[132]:





df['Date']=pd.to_datetime(df['Date'], format='%m%d%Y', errors='ignore')





df['Yesterday'] = pd.to_datetime(df.Date) + pd.to_timedelta(-1, unit="D")





confirmed_2.rename(columns={'Date':'Yesterday'}, inplace=True)

deaths_2.rename(columns={'Date':'Yesterday'}, inplace=True)

recovered_2.rename(columns={'Date':'Yesterday'}, inplace=True)



confirmed_2['Yesterday']=pd.to_datetime(confirmed_2['Yesterday'], format='%m%d%Y', errors='ignore')

deaths_2['Yesterday']=pd.to_datetime(deaths_2['Yesterday'], format='%m%d%Y', errors='ignore')

recovered_2['Yesterday']=pd.to_datetime(recovered_2['Yesterday'], format='%m%d%Y', errors='ignore')



confirmed_2['Yesterday'] = pd.to_datetime(df.Date) + pd.to_timedelta(0, unit="D")

deaths_2['Yesterday'] = pd.to_datetime(df.Date) + pd.to_timedelta(0, unit="D")

recovered_2['Yesterday'] = pd.to_datetime(df.Date) + pd.to_timedelta(0, unit="D")



df=pd.merge(df, confirmed_2[['Province/State','Country/Region','Yesterday','value']], on=['Yesterday','Country/Region','Province/State'] )

df.rename(columns={'value':'Confirmed_y'}, inplace=True)



df=pd.merge(df, deaths_2[['Province/State','Country/Region','Yesterday','value']], on=['Yesterday','Country/Region','Province/State'] )

df.rename(columns={'value':'Deaths_y'}, inplace=True)



df=pd.merge(df, recovered_2[['Province/State','Country/Region','Yesterday','value']], on=['Yesterday','Country/Region','Province/State'] )

df.rename(columns={'value':'Recovered_y'}, inplace=True)





df['Confirmed_var']=df['Confirmed']-df['Confirmed_y']

df['Deaths_var']=df['Deaths']-df['Deaths_y']

df['Recovered_var']=df['Recovered']-df['Recovered_y']





df=df[['Province/State', 'Country/Region', 'Lat', 'Long', 'Date', 'Confirmed',

       'Deaths', 'Recovered', 'Mortality_Rate(D/C)',

       'Confirmed_var', 'Deaths_var', 'Recovered_var']]



df['Date'] = pd.to_datetime(df.Date) + pd.to_timedelta(0, unit="D")





# # Outuput



# In[133]:





df.to_csv('COVID_19.csv')
