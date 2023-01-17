# Ensure plotly==3.10.0 Newer versions have a BUG affecting fbprophet



import pandas as pd

import numpy as np



import datetime

import holidays



from fbprophet import Prophet

from fbprophet.make_holidays import make_holidays_df

from fbprophet.plot import add_changepoints_to_plot





from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize']=(20,10)

#plt.style.use('fivethirtyeight')

plt.style.use('ggplot')
#function to remove any negative forecasted values.

def remove_negs(ts):

    ts['yhat'] = ts['yhat'].clip(lower=0)

    ts['yhat_lower'] = ts['yhat_lower'].clip(lower=0)

    ts['yhat_upper'] = ts['yhat_upper'].clip(lower=0)
# Define dates to exclude i.e. holidays and weekends

# TO-DO: Figure out Eid al-Fitr and Eid al-Adha



# Holidays not built-in e.g. idd and declared holidays

#holidays = pd.DataFrame({

#    'holiday' : 'holiday',

#    'ds': pd.date_range(start='2019-12-25',end='2019-12-26'),

#})



year_list = [2014, 2015, 2016, 2017, 2018, 2019, 2020]

holidays = make_holidays_df(year_list=year_list, country='KE')



# Get country holidays

#holidays = pd.DataFrame(holidays.Kenya(years = 2019).items(), columns=['ds', 'holiday'])

holidays = holidays.append([{'holiday' : 'Moi Day', 'ds' : pd.to_datetime('2018-10-10')}])

holidays = holidays.append([{'holiday' : 'Moi Day', 'ds' : pd.to_datetime('2019-10-10')}])

holidays = holidays.append([{'holiday' : 'Moi Day', 'ds' : pd.to_datetime('2020-10-10')}])

#holidays['ds'] = pd.to_datetime(holidays['ds'])

#print(holidays)





#weekends = pd.DataFrame({

#    'holiday': 'weekend',

#    'ds': pd.to_datetime(['2019-12-14','2019-12-15','2019-12-21','2019-12-22','2019-12-28','2019-12-29',

#                         '2020-01-04','2020-01-05','2020-01-11','2020-01-12']),

#})



#combine dataframes into one - concatenation of holidays after removing duplicates

exclude_days = holidays #pd.concat([holidays, weekends], ignore_index=True)

exclude_days.drop_duplicates(keep='first')

exclude_days = exclude_days.sort_values(by='ds')

#print(exclude_days.tail())
ticker = 'KCB'

predict_days = 30

history_5years = 1305  # Use days == Last 5 calendar years data excluding weekends (365*5 - 52*2*5 == 1825-520 = 1305)



data_folder = 'data/'

data_file = data_folder+ticker+'.csv'







data = pd.read_csv(data_file, index_col='Date', parse_dates=True)

data.drop(['Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1, inplace=True)



data = data.tail(history_5years) #exclude 52 weeks weekends for 5 years

data = data.dropna() # remove blank/missing values

#2014-09-25  14.45



#print(data.head())

#print(data.shape)

#data.info()



data.plot()
df = data.reset_index().rename(columns={'Date':'ds', 'Close':'y'})

df['y'] = np.log(df['y'])



#holidays=holidays #interval_width=0.95 confidence interval default == 80%

#seasonality_mode='multiplicative'



model = Prophet(interval_width=0.95, changepoint_prior_scale=0.01) 



model.fit(df)



future = model.make_future_dataframe(periods=predict_days, freq = 'd') #forecasting for 1 year from now.

future = future[~future['ds'].isin(exclude_days['ds'])] #Exclude days in forecast

future = future[future["ds"].apply(lambda x: x.weekday())<5] #Filter out weekends in predictions

#print(future.tail())



forecast = model.predict(future)



remove_negs(forecast)

#print(future)

figure=model.plot(forecast)
model.plot_components(forecast);
fig_changes=model.plot(forecast)

a=add_changepoints_to_plot(fig_changes.gca(),model,forecast)
two_years = forecast.set_index('ds').join(data)

two_years = two_years[['Close', 'yhat', 'yhat_upper', 'yhat_lower' ]].dropna().tail(history_5years) #Last 3(1095) years

two_years['yhat']=np.exp(two_years.yhat)

two_years['yhat_upper']=np.exp(two_years.yhat_upper)

two_years['yhat_lower']=np.exp(two_years.yhat_lower)



#print(forecast.tail())

print(two_years.tail())

two_years[['Close', 'yhat']].plot()
#'average error'

two_years_AE = (two_years.yhat - two_years.Close)

print (two_years_AE.describe())
# R-squared / coefficient of determination

print('R2 SCORE == '+str(round(r2_score(two_years.Close, two_years.yhat),4)))

print('     MSE == '+str(round(mean_squared_error(two_years.Close, two_years.yhat),4))) #for MSE, closer to zero is better

print('     MAE == '+str(round(mean_absolute_error(two_years.Close, two_years.yhat),4)))


fig, ax1 = plt.subplots()

ax1.plot(two_years.Close)

ax1.plot(two_years.yhat)

ax1.plot(two_years.yhat_upper, color='black',  linestyle=':', alpha=0.5)

ax1.plot(two_years.yhat_lower, color='black',  linestyle=':', alpha=0.5)



ax1.set_title('Actual '+ticker+' (Orange) vs Forecasted Upper & Lower Confidence (Black)')

ax1.set_ylabel('Share Price')

ax1.set_xlabel('Date')
full_df = forecast.set_index('ds').join(data)

full_df['yhat']=round(np.exp(full_df['yhat']),2)

full_df['yhat_upper']=round(np.exp(full_df['yhat_upper']),2)

full_df['yhat_lower']=round(np.exp(full_df['yhat_lower']),2)


fig, ax1 = plt.subplots()

ax1.plot(full_df.Close)

ax1.plot(full_df.yhat, color='black', linestyle=':')

ax1.fill_between(full_df.index, full_df['yhat_upper'], full_df['yhat_lower'], alpha=0.5, color='darkgray')

ax1.set_title('[ '+ticker+' ] - Actual (Orange) vs Forecasted (Black) with Confidence Bands')

ax1.set_ylabel('Share Price')

ax1.set_xlabel('Date')



L=ax1.legend() #get the legend

L.get_texts()[0].set_text('Actual') #change the legend text for 1st plot

L.get_texts()[1].set_text('Forecasted') #change the legend text for 2nd plot
print('Forecast Data -------------------------------------------')

#full_df.tail()



print (full_df[['yhat', 'yhat_lower', 'yhat_upper']].tail(predict_days))