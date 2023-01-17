import pandas as pd

from fbprophet import Prophet
df = pd.read_csv('../input/coronavirus-dataset-02292020/2019_nCoV_data_03-06-2020.csv',parse_dates=['Last Update'])

df.info() 
df["Confirmed"].fillna(0.0, inplace = True)

df["Deaths"].fillna(0.0, inplace = True)

df["Recovered"].fillna(0.0, inplace = True)

df.info()
country_list = df['Country/Region'].unique()

for country in sorted(country_list):

    print('- {}'.format(country))
all_countries = df['Country/Region'].unique()

len(all_countries)
df.loc[df['Country/Region'] == 'Mainland China', 'Country'] = 'China'

df['Last Update'] = pd.to_datetime(df['Last Update'])

df['Date'] = df['Last Update'].apply(lambda x:x.date())

df_by_date=df.groupby(['Date']).sum().reset_index(drop=None)

df_by_date
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()
confirmed.columns = ['ds','y']

#confirmed['ds'] = confirmed['ds'].dt.date

confirmed['ds'] = pd.to_datetime(confirmed['ds'])



deaths.columns = ['ds','y']

#deaths['ds'] = deaths['ds'].dt.date

deaths['ds'] = pd.to_datetime(deaths['ds'])



recovered.columns = ['ds','y']

#recovered['ds'] = recovered['ds'].dt.date

recovered['ds'] = pd.to_datetime(recovered['ds'])
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=21)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m = Prophet(interval_width=0.95)

m.fit(deaths)

future = m.make_future_dataframe(periods=21)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m = Prophet(interval_width=0.95)

m.fit(recovered)

future = m.make_future_dataframe(periods=21)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
confirmed = df.groupby(['Country/Region', 'Date']).sum()['Confirmed'].reset_index()
confirmed[confirmed['Country/Region'] == 'Italy']