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
recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

death = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

cv_opline = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

cv_linelist = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

cv_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
import matplotlib.pyplot as plt
country = 'China'
# Select country by /country/ var



select = confirmed['Country/Region'] == country



# Copy filtered dataFrame to temporary dataFrame



confirmed_s = confirmed[select]

death_s = death[select]

recovered_s = recovered[select]





# Transpose dataFrame using sum()

## Menggunakan sum, karena ingin melihat Country secara keseluruhan tidak mendetail sampai Province



confirmed_st = pd.DataFrame(confirmed_s.iloc[:,4:].sum()).rename(columns={0:'Confirmed'})

death_st = pd.DataFrame(death_s.iloc[:,4:].sum()).rename(columns={0:'Death'})

recovered_st = pd.DataFrame(recovered_s.iloc[:,4:].sum()).rename(columns={0:'Recovered'})



# Merge Confirmed, Recovered, and Death dataFrames filtered by country to single dataFrame



country_st = pd.merge(pd.merge(confirmed_st, death_st, left_index=True, right_index=True),

                      recovered_st, left_index=True, right_index=True)



# Set the index type to datetime



# country_st.index = pd.to_datetime(country_st.index) 
plt.figure(figsize=[30,10])

plt.title('{} Covid-19 Confirmed/Recovered/Death Cases'.format(country))

plt.plot(country_st['Confirmed'], color='blue')

plt.plot(country_st['Recovered'], color='green')

plt.plot(country_st['Death'], color='red')

plt.xticks(rotation=45)

# plt.yscale('log')

plt.show()
fig, ax = plt.subplots(3, figsize=(30,10))

ax[0].plot(country_st['Confirmed'], color='blue')

ax[1].plot(country_st['Recovered'], color='green')

ax[2].plot(country_st['Death'], color='red')

fig.autofmt_xdate(rotation=45)

plt.show()
country_st
from fbprophet import Prophet
df_confirmed = country_st.Confirmed.reset_index()

df_confirmed.rename(columns={'index': 'ds', 'Confirmed': 'y'}, inplace=True)



df_recovered = country_st.Recovered.reset_index()

df_recovered.rename(columns={'index': 'ds', 'Recovered': 'y'}, inplace=True)



df_death = country_st.Death.reset_index()

df_death.rename(columns={'index': 'ds', 'Death': 'y'}, inplace=True)
df_recovered
# Fit dataFrame to model



m_confirmed = Prophet()

m_confirmed.fit(df_confirmed)

future_confirmed = m_confirmed.make_future_dataframe(periods=30)

forecast_confirmed = m_confirmed.predict(future_confirmed)

forecast_confirmed[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m_recovered = Prophet()

m_recovered.fit(df_recovered)

future_recovered = m_recovered.make_future_dataframe(periods=30)

forecast_recovered = m_confirmed.predict(future_recovered)

forecast_recovered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m_death = Prophet()

m_death.fit(df_death)

future_death = m_death.make_future_dataframe(periods=30)

forecast_death = m_confirmed.predict(future_death)

forecast_death[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m_confirmed.plot(forecast_confirmed)

fig2 = m_recovered.plot(forecast_recovered)

fig3 = m_death.plot(forecast_death)
# Forecastingnya masih pr