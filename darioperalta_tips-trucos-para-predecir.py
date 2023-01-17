import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') 
humidity = pd.read_csv('../input/historical-hourly-weather-data/humidity.csv', index_col='datetime', parse_dates=['datetime'])

humidity.head()
humidity["Vancouver"].asfreq('M').plot() #Frecuencia Mensual

plt.show()
humidity.loc[humidity.index.year==2015]["Vancouver"].asfreq('D').plot()

plt.show()
humidity.loc[(humidity.index.weekday==0) & (humidity.index.year==2013)

            ]["Vancouver"].plot() # Seleccionando solo los lunes

plt.show()
humidity.loc["2013-01-01":"2013-01-15"]["Vancouver"].plot()

plt.show()
# Forma para ver % de datos faltantes por columna.

(humidity.isna().sum(axis=0) / humidity.shape[0] * 100)
humidity = humidity.fillna(method='ffill')

humidity.head()
humidity = humidity.fillna(method='bfill')

humidity.head()
humidity["Vancouver"].asfreq('M').head()
humidity.groupby(pd.Grouper(freq='M'))["Vancouver"].mean().head()
humidity["Los Angeles"].tail()
humidity["Los Angeles"].shift(1).tail()
humidity["Vancouver"].asfreq('M').plot(legend=True)

shifted = humidity["Vancouver"].asfreq('M').shift(10).plot(legend=True)

shifted.legend(['Vancouver','Vancouver_lagged'])

plt.show()
humidity["Vancouver"].asfreq('M').plot(legend=True)

shifted = humidity["Vancouver"].asfreq('M').shift(-10).plot(legend=True)

shifted.legend(['Vancouver','Vancouver_lagged'])

plt.show()
humidity["Los Angeles"].tail()
humidity["Los Angeles"].diff(1).tail()
humidity["Los Angeles"].pct_change(1).tail() * 100
rolling_los_angeles = humidity["Los Angeles"].asfreq('M').rolling(3).mean() # Promedio de los ultimos 3 periodos

humidity["Los Angeles"].asfreq('M').plot()

rolling_los_angeles.plot()

plt.legend(['Regular','Rolling Mean'])

plt.show()
expanding_los_angeles = humidity["Los Angeles"].asfreq('M').expanding().mean() # Promedio de los ultimos 3 periodos

humidity["Los Angeles"].asfreq('M').plot()

expanding_los_angeles.plot()

plt.legend(['Regular','Expanding Mean'])

plt.show()
### Convertir datos a formato longitudinal.

humidity_long = humidity.reset_index()

humidity_long = pd.melt(humidity_long,id_vars='datetime')

humidity_long.columns = ["datetime","area","humidity"]

humidity_long.head()
humidity_long["change"] = humidity_long.groupby("area")["humidity"].pct_change()

humidity_long.head(10)