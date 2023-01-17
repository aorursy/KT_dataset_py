import pandas as pd



df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

df.Date = pd.to_datetime(df.Date)

df.head()
total = df.groupby(['Date']).sum().loc[:,['Confirmed','Deaths','Recovered']].reset_index()



total.head()
import fbprophet

from fbprophet.plot import add_changepoints_to_plot



df_prophet= total.rename(columns={'Date': 'ds', 'Confirmed': 'y'})



# Make a future dataframe for X days

m_global = fbprophet.Prophet(changepoint_prior_scale=0.05,changepoint_range=0.95,

                      daily_seasonality=False, 

                      weekly_seasonality=True,

                     mcmc_samples=300)

# Add seasonlity

m_global.add_seasonality(name='monthly', period=30.5, fourier_order=5)



m_global.fit(df_prophet)



# Make predictions

future_global = m_global.make_future_dataframe(periods=30, freq='D')



forecast_global = m_global.predict(future_global)



m_global.plot_components(forecast_global);

fig =m_global.plot(forecast_global)
fig = m_global.plot(forecast_global)

a = add_changepoints_to_plot(fig.gca(), m_global, forecast_global)
# restrict to one country

df_china = df[df['Country/Region']=='Mainland China']

total_china = df_china.groupby(['Date']).sum().loc[:,['Confirmed','Deaths','Recovered']].reset_index()

total_china.head()
china_prophet= total_china.rename(columns={'Date': 'ds', 'Confirmed': 'y'})



# Make a future dataframe for X days

m_china = fbprophet.Prophet(changepoint_prior_scale=0.05,changepoint_range=0.95,

                      daily_seasonality=False, 

                      weekly_seasonality=True,

                     mcmc_samples=300)

# Add seasonlity

m_china.add_seasonality(name='monthly', period=30.5, fourier_order=5)



m_china.fit(china_prophet)



# Make predictions

future_china = m_china.make_future_dataframe(periods=30, freq='D')



forecast_china = m_china.predict(future_china)



m_china.plot_components(forecast_china);
import seaborn as sns

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(12,6))



fig = sns.set_palette('viridis')

fig = sns.lineplot(x='Date', y='Confirmed',data = total, label='Global')

fig = sns.lineplot(x='Date', y='Confirmed',data = total_china, label='China')



sns.despine()

plt.legend()

plt.tight_layout()

plt.xticks(rotation=40)
print('Percentage of global cases in China: %s' %((total_china.Confirmed.sum()*100)/total.Confirmed.sum()))
fig = m_china.plot(forecast_china)
forecast_china[len(total_china):].loc[:,['ds', 'yhat_lower' ,'yhat_upper', 'yhat']].iloc[:7]
forecast_global[len(total):].loc[:,['ds', 'yhat_lower' ,'yhat_upper', 'yhat']].iloc[:7]