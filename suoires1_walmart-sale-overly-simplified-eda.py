import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()

import os
# set working directory to input folder

path = "/kaggle/input/m5-forecasting-accuracy/"

try:

    os.chdir(path)

    print("Directory changed to:", path)

except OSError:

    print("Can't change the Current Working Directory")

    

# import data

sell_prices = pd.read_csv("sell_prices.csv")

calendar = pd.read_csv('calendar.csv')

sales = pd.read_csv('sales_train_validation.csv')

sample_submission = pd.read_csv('sample_submission.csv')



print("Data loaded")
sample_submission.head()
validation_prop = len(sample_submission[sample_submission['id'].str.contains('evaluation')]) / len(sample_submission)



print("Proportion of validation rows in sample_submission.csv:", validation_prop)
# gather a bunch of dates for timeline

sell_prices_cal = pd.merge(sell_prices, calendar, how = 'left', on = 'wm_yr_wk')





calendar_startdate = calendar.date.min()

calendar_enddate = calendar.date.max()

sales_train_validation_startdate = calendar[calendar.d == 'd_1'].date.item()

sales_train_validation_enddate = calendar[calendar.d == 'd_1913'].date.item()

submission_validation_startdate = calendar[calendar.d == 'd_1914'].date.item()

submission_validation_enddate = calendar[calendar.d == 'd_1941'].date.item()

submission_evaluation_startdate = calendar[calendar.d == 'd_1942'].date.item()

submission_evaluation_enddate = calendar[calendar.d == 'd_1969'].date.item()

sell_price_startdate = sell_prices_cal.date.min()

sell_price_enddate = sell_prices_cal.date.max()



del sell_prices_cal
import plotly.figure_factory as ff



df = [dict(Task="Sell Prices", Start = sell_price_startdate, Finish = sell_price_enddate),

      dict(Task="Calendar", Start = calendar_startdate, Finish = calendar_enddate),

      dict(Task="Sales train validation", Start = sales_train_validation_startdate, Finish = sales_train_validation_enddate),

      dict(Task="Submission validation", Start = submission_validation_startdate, Finish = submission_validation_enddate),

      dict(Task="Submission evaluation", Start = submission_evaluation_startdate, Finish = submission_evaluation_enddate)]



fig = ff.create_gantt(df)

fig.show()
sell_prices.head()
sell_prices['store_id'].value_counts()
# using vectorized str.split will be much faster than using apply here

sell_prices['state'] =  sell_prices['store_id'].str.split('_').str[0]
sell_prices['state'].unique().tolist()
sell_prices['product_cat'] =  sell_prices['item_id'].str.split('_').str[0]

sell_prices['product_cat'].unique().tolist()
from statsmodels.graphics.mosaicplot import mosaic



mosaic(sell_prices, ['state', 'product_cat'])

plt.show()
sell_prices['state'].value_counts().plot(kind = 'bar')

plt.title('Number of rows by State')

plt.show()
sell_prices['store_id'].value_counts()

sell_prices.groupby(['state'])['sell_price'].mean()
sell_prices.groupby(['product_cat'])['sell_price'].mean()
plt.figure(figsize=(18,9)) # ah.. the sweet 18 by 9 ratio



sns.lineplot(x = 'wm_yr_wk', y = 'sell_price', hue = 'product_cat', data = sell_prices)

plt.title("Sell Price of Product Categories over time")

plt.show()
plt.figure(figsize=(18,9))



sns.lineplot(x = 'wm_yr_wk', y = 'sell_price', hue = 'state', data = sell_prices)

plt.title("Sell Price of Product Categories over time")

plt.show()
calendar.head()
# duplicate check

calendar['d'].duplicated().sum()
# convert date column (str) to date_time object

calendar['date'] = pd.to_datetime(calendar['date'])



# date range

display(max(calendar['date']) - min(calendar['date']))



# last d value

display(calendar['d'].tail(1))
display(calendar['event_name_1'].value_counts())

display(calendar['event_name_2'].value_counts())
plt.figure(figsize=(18,9))

sns.heatmap(calendar.isnull(), cbar = False)

plt.show()
# before filling NaNs, better check if other columns contains any NaNs

calendar.isnull().sum(axis = 0)
calendar = calendar.fillna("Normal")

calendar.head()
events_weekly = calendar[calendar['event_type_1'] != 'Normal'][['wm_yr_wk', 'event_name_1', 'event_type_1']]

display(events_weekly)

print("Number of week duplicates:", events_weekly['wm_yr_wk'].duplicated().sum())
# "Some of You May Die, but that is a Sacrifice I'm Willing to Make"

events_weekly = events_weekly.drop_duplicates(subset = 'wm_yr_wk', keep = 'first')
# merge + fill NaNs

price_with_event = pd.merge(sell_prices, events_weekly, how = 'left', on = ['wm_yr_wk']).fillna('Normal')



# new column denotes event

price_with_event['event'] = np.where(price_with_event['event_type_1'] == 'Normal', 'Normal', 'Event')

price_with_event.head()
plt.figure(figsize=(18,9))



sns.relplot(x = 'wm_yr_wk', y = 'sell_price', 

            hue = 'event_type_1', style = 'event_type_1', row = 'product_cat', 

            height = 4, # make the plot 4 units high

            aspect = 3, # height should be three times width

            kind = 'line',data = price_with_event, ci = None)  # remove confident interval for better clarity



plt.title("Sell Price of Product Categories over time")

plt.show()
sns.boxplot(x = 'event_type_1', y = 'sell_price', data = price_with_event)

plt.show()
temp_df = price_with_event.groupby(['item_id', 'event'])['sell_price'].min().unstack()

temp_df.head()
temp_df['event_delta'] = temp_df['Event'] - temp_df['Normal']

temp_df.head()
sum(temp_df['event_delta'] < 0) / len(temp_df)
temp_df['discount_prop'] = temp_df['event_delta'] / temp_df['Normal']

temp_df
temp_df.sort_values(by=['discount_prop']).head(10)
# reset the index to get the column item_id back

temp_df.reset_index(inplace = True)



# get item department from item_id

temp_df['department'] = temp_df['item_id'].str.split('_').str[:2].apply(lambda x: '_'.join(x))

temp_df.head()

temp_df[temp_df['discount_prop'] < 0].groupby(['department'])['discount_prop'].count().sort_values(ascending = False).plot(kind = 'bar')

plt.show()
sales.head()
sales_long = pd.melt(sales, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'd', value_name = 'unit_sold')

sales_long.head()
sales_cat = sales_long.groupby(['d', 'cat_id'])['unit_sold'].sum().unstack()

sales_cat
# reset index

sales_cat.reset_index(inplace = True)



# melt the cat

sales_cat = pd.melt(sales_cat, id_vars =['d'], var_name = 'product_cat', value_name = 'unit_sold')



# merge with calendar to get datetime

sales_cat = pd.merge(sales_cat, calendar[['date', 'd']], how = 'left', on = 'd')



sales_cat
plt.figure(figsize=(18,9))



sns.lineplot(x = 'date', y = 'unit_sold', hue = 'product_cat', data = sales_cat)

plt.title("Units Sold for each Product Category")

plt.show()
sales_cat[sales_cat['unit_sold'] < 500]
sales_univariate = sales_long.groupby(['d'])['unit_sold'].sum().reset_index()



# merge with calendar to get datetime

sales_univariate = pd.merge(sales_univariate, calendar[['date', 'd', 'weekday', 'month', 'year']], how = 'left', on = 'd')



sales_univariate.head()
sales_univariate.groupby(['weekday'])['unit_sold'].sum().plot(kind = 'bar')

plt.title("Number of Unit Sold by Weekday")

plt.show()
sales_univariate.groupby(['month'])['unit_sold'].sum().plot(kind = 'line')

plt.title("Number of Unit Sold by Month")

plt.show()
sales_univariate.groupby(['year'])['unit_sold'].sum().plot(kind = 'line')

plt.title("Number of Unit Sold by Year")

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose



# set date as index

sales_univariate = sales_univariate.set_index('date')



# get the unit_sold series

unit_sold_series = sales_univariate['unit_sold'].sort_index()
plt.figure(figsize = (16,10))

# period is the number of observations in a series if you consider the natural time interval of measurement (weekly, monthly, yearly)

decomposition = seasonal_decompose(unit_sold_series, model='multiplicative',period= 365) 

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(unit_sold_series, label = 'Original')

plt.legend(loc = 'best')

plt.subplot(412)

plt.plot(trend, label = 'Trend')

plt.legend(loc = 'best')

plt.subplot(413)

plt.plot(seasonal, label = 'Seasonal')

plt.legend(loc = 'best')

plt.subplot(414)

plt.plot(residual, label = 'Residuals')

plt.legend(loc = 'best')

plt.tight_layout()
holiday_lists = calendar[(calendar['date'].dt.year == 2013) & (calendar['event_name_1'] != 'Normal')][['date', 'event_name_1']]

holiday_lists = pd.merge(holiday_lists, seasonal.reset_index())

holiday_lists['dayofyear'] = holiday_lists.date.dt.dayofyear



plt.figure(figsize = (16,10))

sns.lineplot(x = seasonal.index.dayofyear, y = seasonal)

p1 = sns.scatterplot(x = holiday_lists.date.dt.dayofyear, y = holiday_lists.seasonal, color = 'red')



# add holoday labels

for line in range(0,holiday_lists.shape[0]):

     p1.text(holiday_lists.dayofyear[line]+0.2, holiday_lists.seasonal[line], holiday_lists.event_name_1[line], horizontalalignment='left', size='medium', color='black')

plt.title('Average Impact of Seasonal and Events on Sales')
# Prepare Data

x = holiday_lists.loc[:, ['seasonal']]

holiday_lists['z'] = x - 1 

holiday_lists['colors'] = ['red' if x < 0 else 'green' for x in holiday_lists['z']]

holiday_lists.sort_values('z', inplace=True)

holiday_lists.reset_index(inplace=True)



# Draw plot

plt.figure(figsize=(20,10), dpi= 80)

plt.hlines(y=holiday_lists.event_name_1, xmin=0, xmax=holiday_lists.z, color = holiday_lists.colors, alpha=0.4, linewidth=5)



for x, y, tex in zip(holiday_lists.z, holiday_lists.event_name_1, holiday_lists.z):

    t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 

                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':14})



# Decorations

plt.gca().set(ylabel='Event', xlabel='Impact on Sale')

plt.title('Diverging Bars of Event Impact', fontdict={'size':20})

plt.grid(linestyle='--', alpha=0.5)
def item_visualizer(item_name):

    specific_item = sales_long[sales_long['item_id'] == item_name][['item_id', 'd', 'unit_sold']]



    # merge with calendar to get datetime

    specific_item = pd.merge(specific_item, calendar[['date', 'd', 'wm_yr_wk']], how = 'left', on = 'd')



    # merge with sell_prices to get pricing

    specific_item = pd.merge(specific_item, sell_prices[['item_id', 'sell_price', 'wm_yr_wk', ]], how = 'left', on = ['item_id', 'wm_yr_wk'])

    

#     # random sampling to reduce plot time

#     specific_item = specific_item.sample(frac = 0.1, random_state = 42)



    plt.figure(figsize = (14,7))

    ax1 = sns.lineplot(x = 'date', y = 'sell_price', data = specific_item, label = 'Sell Price', color = 'red', alpha = 0.8)

    ax1.legend(loc="upper right")



    ax2 = plt.twinx()

    sns.lineplot(x = 'date', y = 'unit_sold', data = specific_item, label = 'Units Sold', alpha = 0.5, ax=ax2)

    ax2.legend(loc="upper left")



    plt.title('Sell Price and Sales of item: ' + item_name)

    plt.show()
item_visualizer('FOODS_1_005')
# split into train and test sets

unit_sold = unit_sold_series.reset_index()



n_train = round(365 * 4.5) # train on 4.5 year worth of data

train = unit_sold[:n_train]

test = unit_sold[n_train:]
plt.figure(figsize = (16,8))

sns.lineplot(x = train.date, y = train.unit_sold, label = 'Train')

sns.lineplot(x = test.date, y = test.unit_sold, label = 'Test')

plt.title('Total Units Sold from All Malmarts')

plt.show()
from statsmodels.tsa.api import ExponentialSmoothing, Holt



prediction_holt = test.copy()

linear_fit = Holt(np.asarray(train['unit_sold'])).fit()

prediction_holt['Holt_linear'] = linear_fit.forecast(len(test))



plt.figure(figsize = (16,8))

plt.plot(train.unit_sold, label = 'Train')

plt.plot(test.unit_sold, label = 'Test')

plt.plot(prediction_holt['Holt_linear'], label = 'Holt Linear Prediction')

plt.legend(loc = 'best')

plt.title('Holt Linear Trend Forcast')

plt.show()
# calculate RMSE to check the accuracy of the model:

from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(test.unit_sold, prediction_holt.Holt_linear))



print(rms)

prediction_holtwinter = test.copy()

fit1 = ExponentialSmoothing(np.asarray(train['unit_sold']), seasonal_periods= 365, trend = 'add', seasonal= 'add', damped = True).fit()

prediction_holtwinter['Holt_Winter'] = fit1.forecast(len(test))



plt.figure(figsize = (16,8))

plt.plot(train['unit_sold'], label = 'Train')

plt.plot(test['unit_sold'], label = 'Test', alpha = 0.6)

plt.plot(prediction_holtwinter.Holt_Winter, label = 'Holt Winters Prediction', alpha = 0.6)

plt.legend(loc = 'best')

plt.show()
rms = sqrt(mean_squared_error(test.unit_sold, prediction_holtwinter.Holt_Winter))



print(rms)
import statsmodels.api as sm





train = unit_sold_series[:n_train]

test = unit_sold_series[n_train:]



y_hat_avg = test.copy()

fit1 = sm.tsa.statespace.SARIMAX(train, order=(2, 1, 4),seasonal_order=(0,1,1,12)).fit()



y_hat_avg['SARIMA'] = fit1.predict(start="2015-07-29", end="2016-04-24", dynamic=True)



plt.figure(figsize=(16,8))

plt.plot(train, label='Train')

plt.plot(test, label='Test')

plt.plot(y_hat_avg['SARIMA'], label='SARIMA')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test, y_hat_avg['SARIMA']))



print(rms)