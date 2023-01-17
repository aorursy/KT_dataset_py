import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mat

import statsmodels.api as sm

from fbprophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.stattools import adfuller
pd.plotting.register_matplotlib_converters()
mat.rcParams.update({'figure.figsize':(20,15), 'font.size': 14})
energy_consumption = pd.read_csv('../input/hourly-energy-consumption/PJME_hourly.csv')
energy_consumption.head()
energy_consumption.dtypes
energy_consumption.info()
energy_consumption['Datetime'] = pd.to_datetime(energy_consumption['Datetime'])
energy_consumption = energy_consumption.set_index('Datetime').resample('H').sum()
energy_consumption.info()
hours_no_consumption = energy_consumption.loc[energy_consumption['PJME_MW'] == 0].copy()
energy_consumption.plot(grid=True)

# plt.yscale('log')

plt.show()
energy_consumption.describe()
energy_consumption.loc[~energy_consumption.index.isin(hours_no_consumption.index)].describe()
energy_consumption.loc[~energy_consumption.index.isin(hours_no_consumption.index)].plot(grid=True)

# plt.yscale('log')

plt.show()
energy_consumption['PJME_MW'].quantile([0.025]).values[0]
energy_consumption['PJME_MW'].quantile([0.975]).values[0]
energy_consumption.loc[(energy_consumption['PJME_MW'] >= energy_consumption['PJME_MW'].quantile([0.025]).values[0])

                      &

                      (energy_consumption['PJME_MW'] <= energy_consumption['PJME_MW'].quantile([0.975]).values[0])].describe()
energy_consumption.loc[(energy_consumption['PJME_MW'] >= energy_consumption['PJME_MW'].quantile([0.025]).values[0])

                      &

                      (energy_consumption['PJME_MW'] <= energy_consumption['PJME_MW'].quantile([0.975]).values[0])].plot(grid=True)

plt.show()
energy_consumption.resample('YS').mean().sort_values('PJME_MW',ascending=False).head(1)
energy_consumption.resample('YS').mean().sort_values('PJME_MW',ascending=False).tail(1)
energy_consumption.resample('YS')[['PJME_MW']].mean().plot(grid=True)

plt.show()
energy_consumption.resample('W').mean().plot(grid=True)

plt.show()
energy_consumption.loc['01-01-2018':].resample('W').mean().sort_values('PJME_MW',ascending=False).head(1)
energy_consumption['Hour'] = energy_consumption.index.hour
max_hour = energy_consumption.loc[energy_consumption.loc['06-01-2018':'07-31-2018'].resample('D')['PJME_MW'].idxmax()].copy()
df = energy_consumption.loc['06-01-2018':'07-31-2018'].resample('D')[['PJME_MW']].sum()

_ = plt.plot(df.index, df.PJME_MW)

for H in max_hour['Hour'].unique():

    df = energy_consumption.loc['06-01-2018':'07-31-2018'].resample('D')[['PJME_MW']].sum()

    df = df.loc[max_hour.loc[max_hour['Hour'] == H].index.date].copy()

    _ = plt.scatter( x = df.index, y = df.PJME_MW, label = H)

plt.grid()

plt.legend()

plt.show()
dayofweek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

energy_consumption['Day of Week'] = energy_consumption.index.dayofweek

energy_consumption['Day of Week'] = energy_consumption['Day of Week'].apply(lambda x: dayofweek[x])
energy_consumption.loc['01-01-2018':].groupby([energy_consumption.loc['01-01-2018':].index.date,

                                               'Day of Week'])[['PJME_MW']].sum().sort_values('PJME_MW',

                                                                                              ascending=False).head()
df = energy_consumption.loc['06-01-2018':'07-31-2018'].resample('D')[['PJME_MW']].sum()

_ = plt.plot(df.index, df.PJME_MW)

for D in energy_consumption['Day of Week'].unique():

    df = energy_consumption.loc['06-01-2018':'07-31-2018'].loc[energy_consumption['Day of Week'] == D].copy()

    df = df.resample('D')[['PJME_MW']].sum().copy()

    df = df.loc[df['PJME_MW'] != 0].copy()

    _ = plt.scatter( x = df.index, y = df.PJME_MW, label = D)

plt.grid()

plt.legend()

plt.show()
energy_consumption.loc['01-01-2017':'31-12-2017'].resample('Q').sum().sort_values('PJME_MW',ascending=False)
year_quarter_con = energy_consumption.copy()

year_quarter_con['Year'] = year_quarter_con.index.year

year_quarter_con['Quarter'] = year_quarter_con.index.quarter
year_quarter_con = year_quarter_con.groupby(['Year','Quarter'])[['PJME_MW']].sum()

year_quarter_con.iloc[year_quarter_con.reset_index().groupby(['Year'])['PJME_MW'].idxmax()]
energy_consumption['Quarter'] = energy_consumption.index.quarter

energy_consumption['Month'] = energy_consumption.index.month
df = energy_consumption.loc[:'31-07-2018'].resample('MS')[['PJME_MW']].sum()

_ = plt.plot( df.index, df.PJME_MW)

for Q in energy_consumption['Quarter'].unique():

    df = energy_consumption.loc[:'31-07-2018'].loc[energy_consumption['Quarter'] == Q].resample('MS')[['PJME_MW']].sum().copy()

    df = df.loc[df['PJME_MW'] != 0].copy()

    _ = plt.scatter( x = df.index, y = df.PJME_MW, label = 'Q' +str(Q))

plt.grid()

plt.legend()

plt.show()
decompose = sm.tsa.seasonal_decompose(energy_consumption.loc[:'31-07-2018'].resample('MS')[['PJME_MW']].sum())

decompose.plot()

plt.show()
day_consum = energy_consumption.loc[:'31-07-2018'].resample('D')[['PJME_MW']].sum()

day_consum.hist(bins=int(np.sqrt(len(day_consum))))
day_consum.reset_index(inplace=True)
day_consum['Datetime'] = pd.to_datetime(day_consum['Datetime'])
day_consum.index = pd.DatetimeIndex(day_consum['Datetime'],freq='D')
day_consum.drop(['Datetime'],1,inplace=True)
result = adfuller(day_consum['PJME_MW'].values)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))

    if result[0] <= value:

        print('Stationary at ' + key)

    else:

        print('Non-Stationary at ' + key)
day_consum['ds'] = day_consum.index

day_consum.rename(columns={'PJME_MW':'y'},inplace=True)
split_date = '06-30-2018'

train = day_consum.loc[:split_date].copy()

test = day_consum.loc[split_date:].copy()
model = Prophet()
model.fit(train)
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)
model.plot(forecast)

plt.show()
model.plot_components(forecast)

plt.show()
prediction_vs_real = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(day_consum.set_index('ds'))
def calculate_forecast_errors(df, prediction_size):

    """Calculate MAPE and MAE of the forecast.

    

       Args:

           df: joined dataset with 'y' and 'yhat' columns.

           prediction_size: number of days at the end to predict.

    """

    

    # Make a copy

    df = df.copy()

    

    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.

    df['e'] = df['y'] - df['yhat']

    df['p'] = 100 * df['e'] / df['y']

    

    # Recall that we held out the values of the last `prediction_size` days

    # in order to predict them and measure the quality of the model. 

    

    # Now cut out the part of the data which we made our prediction for.

    predicted_part = df[-prediction_size:]

    

    # Define the function that averages absolute error values over the predicted part.

    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

    

    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.

    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}
calculate_forecast_errors(prediction_vs_real, len(test))