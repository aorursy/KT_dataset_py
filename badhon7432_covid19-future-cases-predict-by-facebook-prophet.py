import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fbprophet import Prophet
covid_df = pd.read_csv('../input/covid19-global/covid19_global.csv')
covid_df.info()
covid_df.isnull().sum()
covid_df.sort_values('Date')
plt.figure(figsize = (10,10))
plt.plot(covid_df['Date'],covid_df['Confirmed'])
covid_df = covid_df[covid_df['Country']=='Bangladesh']
covid_df = covid_df[['Date','Confirmed']]
covid_df = covid_df.rename(columns = {'Date':'ds', 'Confirmed':'y'})
covid_df.head()
m = Prophet()
m.fit(covid_df)
future = m.make_future_dataframe(periods=60)  # prediction for next 60 days
forecast = m.predict(future)
# visualize the result
figure = m.plot(forecast, xlabel = 'Date', ylabel = 'No of Cases')
plt.title('Covid-19 Cases Forecasting - Bangladesh')
# check the data 
forecast.head()