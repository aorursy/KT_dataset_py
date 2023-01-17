import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet

from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('../input/SeoulHourlyAvgAirPollution.csv')
df.columns = ['Date', 'Location', 'NO2', 'O3', 'CO', 'SO2', 'Fine Dust', 'Ultrafine Dust']

df.head(3)
df.info()
df.dropna(inplace=True)

pollutant_columns = ['NO2', 'O3', 'CO', 'SO2', 'Fine Dust', 'Ultrafine Dust']

df['total_pollutants'] = 0

for col in pollutant_columns:

    df[f'scaled_{col}'] = pd.np.ravel(MinMaxScaler().fit_transform(df[col].values.reshape(-1, 1)))

    df['total_pollutants'] += df[f'scaled_{col}']

# ugly hack to get convert timestamp into a proper datetime

df['Date'] = pd.to_datetime(df.Date.apply(lambda x: str(x)[:-4] + ' ' + str(int(str(x)[-4:])/100) + ':00'))
df.head(3)
ts_data = df.groupby(['Date']).agg('mean').reset_index()[['Date', 'total_pollutants']]

ts_data.rename(columns={'Date': 'ds', 'total_pollutants': 'y'}, inplace=True)

m = Prophet()

m.fit(ts_data)

future = m.make_future_dataframe(periods=24, freq='H')

forecast = m.predict(future)

m.plot(forecast);