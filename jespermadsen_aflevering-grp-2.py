# initial imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'], usecols=['date','d'])



# udvælger kun de datoer som ligger i sales_train_validation



calendar_stv = calendar_df[:1913] 

calendar_stv.info()

del calendar_df
sales_train_validation = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv', index_col='id')

sales_train_validation.head()
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [10, 5]

ax = sales_train_validation.groupby(['store_id'])['cat_id'].value_counts().plot(kind='bar', title="observationer i datasæt fordelt på store og kategory")

ax.set_ylabel('# observationer')

ax.set_xlabel('Store - kategory')



plt.show()
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [6, 2.5]

ax = sales_train_validation.groupby('state_id')['store_id'].nunique().plot(kind='bar', title="Butikker per stat")

ax.set_ylabel('# butikker')

ax.set_xlabel('Stat')

ax.set_ylim(bottom=2)

plt.yticks(np.arange(2,5,1))



plt.show()
aggregate_state_sum = sales_train_validation.groupby(by=['state_id'],axis=0).sum()

aggregate_state_sum.columns = calendar_stv['date']

agg_state_sum_trans = aggregate_state_sum.transpose()

del aggregate_state_sum
from_year = '2015'

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [20, 10]

plt.rcParams['lines.linewidth'] = 2

ax = agg_state_sum_trans[from_year:].plot(title="Summeret salg per stat fra {}".format(from_year))

ax.set_ylabel('Solgte enheder')

plt.show()

del agg_state_sum_trans
aggregate_state_mean = sales_train_validation.groupby(by=['state_id'],axis=0).mean()

aggregate_state_mean.columns = calendar_stv['date']

agg_state_mean_trans = aggregate_state_mean.transpose()

del aggregate_state_mean

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [20, 10]

plt.rcParams['lines.linewidth'] = 2

ax = agg_state_mean_trans['2015':].plot(title="Gennemsnitlig salg per stat")

ax.set_ylabel('Solgte enheder')

plt.show()

del agg_state_mean_trans


aggregate_state_mean = sales_train_validation.groupby(by=['state_id', 'store_id'],axis=0).mean()

aggregate_state_mean.columns = calendar_stv['date']

agg_state_mean_trans = aggregate_state_mean.transpose()

del aggregate_state_mean

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [25, 20]

plt.rcParams['lines.linewidth'] = 2

fig,ax = plt.subplots(3,1)

for i, state in enumerate(['CA','TX','WI'], start=0):

    ax[i].plot(agg_state_mean_trans['2015':][state])

    ax[i].set_title("Gennemsnitlig salg per butik i {}".format(state))

    ax[i].set_ylabel('Solgte enheder')

    i = i+1

plt.show()

del agg_state_mean_trans
aggregate_state_category = sales_train_validation.groupby(by=['state_id', 'cat_id'],axis=0).sum()

aggregate_state_category.columns = calendar_stv['date']

agg_state_trans = aggregate_state_category.transpose()

del aggregate_state_category

fig,ax = plt.subplots(3,1)

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [30, 25]

plt.rcParams['lines.linewidth'] = 2

#ax.legend()

ax[0].plot(agg_state_trans['CA']['2015':])

ax[0].set_title('CA')

ax[0].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')

ax[1].plot(agg_state_trans['TX']['2015':])

ax[1].set_title('TX')

ax[1].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')

ax[2].plot(agg_state_trans['WI']['2015':])

ax[2].set_title('WI')

ax[2].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')

plt.show()

del agg_state_trans


def plot_state_category(sales_train_validation, calendar_dates, state, category='ALL', start_time='2015'):

    sales_state_category = sales_train_validation.loc[sales_train_validation['state_id'] == state ]

    if category != 'ALL' :

        sales_state_category = sales_state_category.loc[sales_state_category['cat_id'] == category]

    aggregate_ssc = sales_state_category.groupby(by=['dept_id'],axis=0).mean()



    aggregate_ssc.columns = calendar_dates['date']



    agg_ssc_trans = aggregate_ssc.transpose()

    plt.style.use('ggplot')

    plt.rcParams['figure.figsize'] = [25, 12]

    plt.rcParams['lines.linewidth'] = 2

    ax = agg_ssc_trans[start_time:].plot(title="MEANed numbers State: {}, Category: {}".format(state, category))

    ax.set_ylabel('Units sold')

    plt.show()

plot_state_category(sales_train_validation, calendar_stv, 'CA', category='FOODS', start_time='2013')
plot_state_category(sales_train_validation, calendar_stv, 'TX', category='FOODS',start_time='2013')
plot_state_category(sales_train_validation, calendar_stv, 'WI', category='FOODS', start_time='2013')
light_sales = sales_train_validation.drop(['item_id','dept_id','cat_id','store_id'], axis=1)

light_sales = light_sales.groupby('state_id').mean()

light_sales.columns = calendar_stv['date']

light_s_t = light_sales.transpose()
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [15, 10]

fig, ax = plt.subplots(2,2)

ax[0][0].plot(light_s_t['20-12-2012':'31-12-2012'])

ax[0][0].set_title('2012')

ax[0][1].plot(light_s_t['20-12-2013':'31-12-2013'])

ax[0][1].set_title('2013')

ax[1][0].plot(light_s_t['20-12-2014':'31-12-2014'])

ax[1][0].set_title('2014')

ax[1][1].plot(light_s_t['20-12-2015':'31-12-2015'])

ax[1][1].set_title('2015')

plt.show()
sales_mean = sales_train_validation.mean()

sales_mean.index = calendar_stv['date']
from scipy import stats

x= np.arange(0,len(sales_mean))

slope, intercept, r_value, p_value, std_err = stats.linregress(x, sales_mean.values)

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [10, 5]

plt.plot(sales_mean.index, sales_mean.values, 'o', label='original data')

plt.plot(sales_mean.index, intercept + slope*x, 'g', label='trend linje')

plt.legend()

plt.show()



store_dept = sales_train_validation.groupby(by= ['cat_id'], axis=0).mean()

store_dept.columns = calendar_stv['date']

store_trans = store_dept.transpose()

del store_dept
weekends = ['01-03-2015','01-04-2015','01-10-2015','01-11-2015','01-17-2015', '01-18-2015','01-24-2015', '01-25-2015', '01-31-2015', 

            '02-01-2015', '02-07-2015', '02-08-2015', '02-14-2015', '02-15-2015', '02-21-2015', '02-22-2015', '02-28-2015', 

            '03-01-2015', '03-07-2015', '03-08-2015', '03-14-2015', '03-15-2015', '03-21-2015', '03-22-2015', '03-28-2015',  '03-29-2015']
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [25, 5]

ax = store_trans['01-01-2015':'04-02-2015'].plot(title="Gns. salg 3 måneder jan-mar 2015")

ax.set_ylabel('# enheder')

ax.vlines(weekends, 0, 2.5, colors=['y','c'])

plt.show()
weekends= ['06-02-2012','06-02-2012','06-09-2012','06-10-2012', '06-16-2012','06-17-2012','06-23-2012', '06-24-2012', '06-30-2012', 

           '07-01-2012','07-07-2012','07-08-2012','07-14-2012', '07-15-2012','07-21-2012','07-22-2012', '07-28-2012', '07-29-2012', 

           '08-04-2012','08-05-2012','08-11-2012', '08-12-2012','08-18-2012','08-19-2012', '08-25-2012', '08-26-2012', '09-01-2012','09-02-2012'

            ]

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [25, 5]

ax = store_trans['06-01-2012':'09-02-2012'].plot(title="Gns. salg 3 måneder jun-aug 2012")

ax.set_ylabel('# enheder')

ax.vlines(weekends, 0, 2.5, colors=['y','c'])

plt.show()

del weekends
#Indlæser data mm.

sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv', index_col='id')

sample = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv', index_col='id')
#De sidste 28 dages salg

lastsales = sales.iloc[:,-28:]
#Laver et array fra "F1" til "F28" som er det påkrævede navneformat for kolonner

f_list = []

for num in np.arange(1,29):

    f_list.append("F{}".format(num))



#Sætter kolonnenavnene for de sidste 28 dages salg til det påkrævede navneformat

lastsales.columns = f_list
#Tilføjer det tomme evalueringsdata til vores salgsdata for at opnå det ønskede format

emptyevaluation = sample.tail(30490)

submission = pd.concat([lastsales, emptyevaluation])



submission.to_csv('baseline.csv')
#De sidste 28 dages salg

lastsales = sales.iloc[:,-28:]
#Vi gør her brug .rolling()-funktionen, som blev introduceret ovenfor.

SMA = lastsales.rolling(window=1, axis=1).mean()
#Laver et array fra "F1" til "F28" som er det påkrævede navneformat for kolonner

f_list = []

for num in np.arange(1,29):

    f_list.append("F{}".format(num))



#Sætter kolonnenavnene for de sidste 28 dages salg til det påkrævede navneformat

SMA.columns = f_list
#Tilføjer det tomme evalueringsdata til vores salgsdata for at opnå det ønskede format

emptyevaluation = sample.tail(30490)

submission = pd.concat([SMA, emptyevaluation])
submission.to_csv('SMA.csv')
#Vi gør her brug .expanding()-funktionen, som blev introduceret ovenfor.

sales_CMA = sales.expanding(min_periods=7, axis=1).mean()
#Vi tager CMA-værdierne for de sidste 28 dage

CMA = sales_CMA.iloc[:,-28:]
#Laver et array fra "F1" til "F28" som er det påkrævede navneformat for kolonner

f_list = []

for num in np.arange(1,29):

    f_list.append("F{}".format(num))



#Sætter kolonnenavnene for de sidste 28 dages salg til det påkrævede navneformat

CMA.columns = f_list
#Tilføjer det tomme evalueringsdata til vores salgsdata for at opnå det ønskede format

emptyevaluation = sample.tail(30490)

submission = pd.concat([CMA, emptyevaluation])
submission.to_csv('CMA.csv')
#Vi gør her brug .ewm()-funktionen, som blev introduceret ovenfor.

sales_EMA = sales.ewm(span=7, axis=1).mean()
#Vi tager EMA-værdierne for de sidste 28 dage

EMA = sales_EMA.iloc[:,-28:]
#Laver et array fra "F1" til "F28" som er det påkrævede navneformat for kolonner

f_list = []

for num in np.arange(1,29):

    f_list.append("F{}".format(num))



#Sætter kolonnenavnene for de sidste 28 dages salg til det påkrævede navneformat

EMA.columns = f_list
#Tilføjer det tomme evalueringsdata til vores salgsdata for at opnå det ønskede format

emptyevaluation = sample.tail(30490)

submission = pd.concat([EMA, emptyevaluation])
submission.to_csv('EMA.csv')
df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'], usecols=['date', 'd'])

sales = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

calendar_stv = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", parse_dates=['date'])

# .mean() fjerner alle kolonner hvor det ikke er muligt at tage et gennemsnit på. 

sales_mean = sales.mean()

sales_mean.index = calendar_stv['date'][:1913]

# Vi transposer for at få vendt dataframen om. Dette gør selve datoen til index. 

sales_mean_trans = sales_mean.transpose()

actual = pd.DataFrame(data=sales_mean_trans)

actual.columns = ['y']

# Vi deler index op i tre kolonner ud fra måned, dag og år. 

actual['month'] = actual.index.month

actual['year'] = actual.index.year

actual['day'] = actual.index.day
# For at vi kan teste hvor godt modellen performer er vi nød til at tage de sidste 28 dage fra træningssættet og gemme de sidste 28 dage i et test set. 

training = actual[:-28]

test = actual[-28:]
from sklearn.ensemble import RandomForestRegressor



y = training['y'].values

X = training.drop('y', axis=1)

X_test = test.drop('y', axis=1)

Y_test = test['y'].values
%%time

rf = RandomForestRegressor()

model = rf.fit(X, y)

y_pred = model.predict(X_test)

y_pred
np.mean((Y_test - y_pred)**2)**0.5
from statsmodels.tsa.statespace.sarimax import SARIMAX
calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'], usecols=['date','d'])

sample_sub = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv', index_col='id')



# udvælger kun de datoer som ligger i sales_train_validation



calendar_stv = calendar_df[:1913]

sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv', index_col='id')

del calendar_df
sales_mean = sales.mean()

sales_mean.index = calendar_stv['date']
x= np.arange(0,len(sales_mean))

slope, intercept, r_value, p_value, std_err = stats.linregress(x, sales_mean.values)

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [10, 5]

plt.plot(sales_mean.index, sales_mean.values, 'o', label='original data')

plt.plot(sales_mean.index, intercept + slope*x, 'g', label='trend linje')

plt.legend()

plt.show()
diffed = sales_mean.diff().dropna()

x= np.arange(0,len(diffed))

slope, intercept, r_value, p_value, std_err = stats.linregress(x, diffed.values)

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [10, 5]

plt.plot(diffed.index, diffed.values, 'o', label='diffed data')

plt.plot(diffed.index, intercept + slope*x, 'g', label='trend linje')

plt.legend()

plt.show()
diffed_season = diffed.diff(7).dropna()

x= np.arange(0,len(diffed_season))

slope, intercept, r_value, p_value, std_err = stats.linregress(x, diffed_season.values)

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = [10, 5]

plt.plot(diffed_season.index, diffed_season.values, 'o', label='diffed data')

plt.plot(diffed_season.index, intercept + slope*x, 'g', label='trend linje')

plt.legend()

plt.show()
from statsmodels.tsa.stattools import adfuller

results = adfuller(sales_mean_trans)

print('ingen dif')

print('  ADF Statistic: {}'.format(results[0]))

print('  p-value: {}'.format(results[1]))

results = adfuller(diffed)

print('diffed')

print('  ADF Statistic: {}'.format(results[0]))

print('  p-value: {}'.format(results[1]))

results = adfuller(diffed_season)

print('diffed+seasonal diffed 7 dage')

print('  ADF Statistic: {}'.format(results[0]))

print('  p-value: {}'.format(results[1]))

results = adfuller(diffed.diff(30).dropna())

print('diffed+seasonal diffed 30 dage')

print('  ADF Statistic: {}'.format(results[0]))

print('  p-value: {}'.format(results[1]))

results = adfuller(diffed.diff(365).dropna())

print('diffed+seasonal diffed 365 dage')

print('  ADF Statistic: {}'.format(results[0]))

print('  p-value: {}'.format(results[1]))
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig,(ax1, ax2) = plt.subplots(2,1, figsize=(20,10))

plot_acf(diffed, zero=False, ax=ax1, lags=21)

plot_pacf(diffed, zero=False, ax=ax2, lags=21)

plt.show()
diffed_7 = diffed.diff(7).dropna()

fig,(ax1, ax2) = plt.subplots(2,1, figsize=(20,10))

plot_acf(diffed_7, zero=False, ax=ax1, lags=21)

plot_pacf(diffed_7, zero=False, ax=ax2, lags=21)

plt.show()
fig,(ax1, ax2) = plt.subplots(2,1, figsize=(20,10))

normal_lags = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

lags7 =  [element * 7 for element in normal_lags]



plot_acf(diffed_7, lags=lags7, ax=ax1)

plot_pacf(diffed_7, lags=lags7, ax=ax2)

plt.show()
%%time

model = SARIMAX(sales_mean, order=(4,1,2), seasonal_order=(5,1,1,7))

results = model.fit()

forecast = results.get_prediction(start=-28)

mean_forecast_is = forecast.predicted_mean

confidence_intervals = forecast.conf_int()



#confidence_intervals

lower_limits = confidence_intervals.loc[:,'lower y']

upper_limits = confidence_intervals.loc[:,'upper y']

plt.figure(figsize=(20,10))

#Plot prediction

plt.plot(sales_mean[-35:].index, sales_mean[-35:].values, label='sales_mean')

plt.plot(mean_forecast_is.index,

         mean_forecast_is.values,

         color='red',

         label='forecast')

#shade uncertainty area

plt.fill_between(mean_forecast_is.index, lower_limits, upper_limits, color='pink')

plt.legend()

plt.grid()

plt.show()
#mean_forecast

RMSE_insample = np.mean((sales_mean[-28:] - mean_forecast_is)**2)**0.5

print("Sarimax(4,1,2)(5,1,1)7\n - RMSE score: {}\n - fitting Tid: {}".format(RMSE_insample, '60 sekunder'))
test = sales_mean[-28:]

train = sales_mean[:-28]
%%time

model = SARIMAX(train, order=(4,1,2), seasonal_order=(5,1,1,7))

results = model.fit()

print(results.aic, results.bic)
print(results.summary())
forecast = results.get_prediction(start='28-03-2016', end='24-04-2016', dynamic=True)

mean_forecast = forecast.predicted_mean

confidence_intervals = forecast.conf_int()



#confidence_intervals

lower_limits = confidence_intervals.loc[:,'lower y']

upper_limits = confidence_intervals.loc[:,'upper y']

plt.figure(figsize=(20,10))

#Plot prediction

plt.plot(train[-20:].index, train[-20:].values, label='Train data')

plt.plot(test.index, test.values, label='Test data')

plt.plot(mean_forecast.index,

         mean_forecast.values,

         color='red',

         label='forecast')

#shade uncertainty area

plt.fill_between(mean_forecast.index, lower_limits, upper_limits, color='pink')

plt.legend()

plt.grid()

plt.show()
#mean_forecast

RMSE = np.mean((test - mean_forecast)**2)**0.5

print("Sarimax(4,1,2)(5,1,1)7\n - RMSE score: {}\n - fitting Tid: {}".format(RMSE, '60 sekunder'))
print('RMSE in sample:     {}\nRMSE out of sample: {}'.format(RMSE_insample, RMSE))
sales = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", parse_dates=['date'])

sales_mean = sales.mean()

sales_mean.index = calendar['date'][:1913]

sales_mean_trans = sales_mean.transpose()
df_ev_1 = pd.DataFrame({'holiday': 'Event 1', 'ds': calendar[~calendar['event_name_1'].isna()]['date']})

df_ev_2 = pd.DataFrame({'holiday': 'Event 2', 'ds': calendar[~calendar['event_name_2'].isna()]['date']})

holidays = pd.concat((df_ev_1, df_ev_2))

del df_ev_1

del df_ev_2

del calendar
ds = sales_mean_trans.index.values

y = sales_mean_trans.values

test = pd.DataFrame(columns=['ds', 'y'])

test['ds'] = ds

test['y'] = y
training = test[:-28]

actual = test[-28:]
%%time

from fbprophet import Prophet

# Vi sender holidays ind som parametre.

model = Prophet(holidays = holidays)

model.fit(training)

forecast = model.make_future_dataframe(periods=28, include_history=False)

forecast = model.predict(forecast)
y_pred = forecast['yhat']

y_actual = actual['y']

y_actual = y_actual.reset_index()['y']
np.mean((y_actual - y_pred)**2)**0.5
score_dict = {

    'naive t-28': [0.10098, 2.0e-3],

    'Moving Average': [1.0942477597466052e-15, 0.0002],

    'SARIMAX': [0.07044, 60],

    'Prophet': [0.08670, 3.47],

    'Random Forrest': [0.35538,0.326]

}



scores = pd.DataFrame(data=score_dict)

scores.index = ['RMSE', 'seconds']

#scores
x = np.arange(3)

fig, (ax,a2) = plt.subplots(2,1,figsize=(10,10))

ax.barh(scores.columns,scores.loc['seconds'])

ax.set_title('Tidsforbrug')

ax.set_xlabel('Sekunder')

a2.barh(scores.columns, scores.loc['RMSE'])

a2.set_title('RMSE')

a2.set_xlabel('Score')

fig.tight_layout(pad=3.0)

plt.show()
