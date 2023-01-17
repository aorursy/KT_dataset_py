import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from numpy import nan

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

from sklearn import metrics

import math





sampleSubmissionData = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

salesTrainValidationData = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sellPricesData = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

calendarData = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
def getWeekdaysInRange(start, end, day):

    dates = []

    for i in range (start,end):

        if calendarData.loc[i].weekday == day:

            dates.append(calendarData.loc[i].date)

    return dates

def getEventDaysInRange(start, end):

    dates = []

    for i in range (start,end):

        if type(calendarData.loc[i].event_name_1) == str or type(calendarData.loc[i].event_name_2) == str:

            dates.append(calendarData.loc[i].date)

    return dates

def getSnapDaysInRange(start, end, state):

    dates = []

    for i in range (start,end):

        if (state == 'CA' and calendarData.loc[i].snap_CA or 

            state == 'TX' and calendarData.loc[i].snap_TX or 

            state == 'WI' and calendarData.loc[i].snap_WI):

            dates.append(calendarData.loc[i].date)

    return dates
salesTrainValidationData.head()
sellPricesData.head()
calendarData.head()
storeCount = salesTrainValidationData.groupby('state_id')['store_id'].nunique().reset_index()

storeCount.columns = ['State', '# of Stores']

storeCount
totalSalesPerDay = salesTrainValidationData.copy().loc[:,"d_1":"d_1913"].sum().to_frame()

dates = pd.to_datetime(calendarData.copy().loc[1:1913].loc[:,'date'].values)

totalSalesPerDay = totalSalesPerDay.set_index(pd.to_datetime(dates))

totalSalesPerDay.plot(title="Total sales per day", figsize=(30,8), legend=None)

plt.style.use('bmh')

plt.xlabel('')

plt.ylabel('Sales')

plt.show()
sales6Months = salesTrainValidationData.copy().loc[:,"d_1733":"d_1913"].sum().to_frame()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

sales6Months = sales6Months.set_index(pd.to_datetime(dates))



sales1Year = salesTrainValidationData.copy().loc[:,"d_1548":"d_1913"].sum().to_frame()

dates = pd.to_datetime(calendarData.copy().loc[1548:1913].loc[:,'date'].values)

sales1Year = sales1Year.set_index(pd.to_datetime(dates))



f,ax = plt.subplots(3,1,figsize=(24,8))

f.tight_layout(pad=3.0)

plt.style.use('bmh')



ax[0].plot(sales6Months, zorder=2)

ax[0].set_title('Total sales per day for the last 6 months with weekends')

ax[0].set_ylabel('Sales')



for date in getWeekdaysInRange(1733, 1913, 'Friday'):

    ax[0].axvline(pd.Timestamp(date), color='#d5dee0', zorder=1, linewidth=8)

    

for date in getWeekdaysInRange(1733, 1913, 'Saturday'):

    ax[0].axvline(pd.Timestamp(date), color='#d5dee0', zorder=1, linewidth=8)

    

for date in getWeekdaysInRange(1733, 1913, 'Sunday'):

    ax[0].axvline(pd.Timestamp(date), color='#d5dee0', zorder=1, linewidth=8)



ax[1].plot(sales1Year, zorder=2)

ax[1].set_title('Total sales per day for the last year with events')

ax[1].set_ylabel('Sales')



for date in getEventDaysInRange(1548, 1913):

    ax[1].axvline(pd.Timestamp(date), color='#cfb4b4', zorder=1, linewidth=8)



ax[2].plot(sales1Year, zorder=2)

ax[2].set_title('Total sales per day for the last year with weekends and events')

ax[2].set_ylabel('Sales')



for date in getWeekdaysInRange(1548, 1913, 'Friday'):

    ax[2].axvline(pd.Timestamp(date), color='#d5dee0', zorder=1, linewidth=8)

    

for date in getWeekdaysInRange(1548, 1913, 'Saturday'):

    ax[2].axvline(pd.Timestamp(date), color='#d5dee0', zorder=1, linewidth=8)

    

for date in getWeekdaysInRange(1548, 1913, 'Sunday'):

    ax[2].axvline(pd.Timestamp(date), color='#d5dee0', zorder=1, linewidth=8)



for date in getEventDaysInRange(1548, 1913):

    ax[2].axvline(pd.Timestamp(date), color='#cfb4b4', zorder=1, linewidth=8)



plt.show()
totalSalesPerDayFirstWinter = salesTrainValidationData.copy().loc[:,"d_1768":"d_1798"].sum().to_frame()

dates = pd.to_datetime(calendarData.copy().loc[1768:1798].loc[:,'date'].values)

totalSalesPerDayFirstWinter = totalSalesPerDayFirstWinter.set_index(pd.to_datetime(dates))



totalSalesPerDaySecondWinter = salesTrainValidationData.copy().loc[:,"d_1403":"d_1433"].sum().to_frame()

dates = pd.to_datetime(calendarData.copy().loc[1403:1433].loc[:,'date'].values)

totalSalesPerDaySecondWinter = totalSalesPerDaySecondWinter.set_index(pd.to_datetime(dates))



f,ax = plt.subplots(2,1,figsize=(20,8))

f.tight_layout(pad=3.0)

plt.style.use('bmh')



ax[0].plot(totalSalesPerDayFirstWinter)

ax[0].set_title('Winter sales 2015')

ax[0].axvline(pd.Timestamp("2015-12-25"), color='#cfb4b4', zorder=1, linewidth=3)



ax[1].plot(totalSalesPerDaySecondWinter)

ax[1].set_title('Winter sales 2014')

ax[1].axvline(pd.Timestamp("2014-12-25"), color='#cfb4b4', zorder=1, linewidth=3)



plt.show()
totalSalesPerDayPast6Months = salesTrainValidationData.copy().groupby('state_id').sum().loc[:,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

totalSalesPerDayPast6Months = totalSalesPerDayPast6Months.set_index(pd.to_datetime(dates))



totalSalesPerDayPast6Months.plot(title="Total sales by states", figsize=(30,8))

plt.style.use('bmh')

plt.show()
dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)



totalSalesPerDayPast6MonthsCA = salesTrainValidationData.copy().groupby('store_id').sum().loc["CA_1":"CA_4", "d_1733":"d_1913"].transpose()

totalSalesPerDayPast6MonthsCA = totalSalesPerDayPast6MonthsCA.set_index(pd.to_datetime(dates))

totalSalesPerDayPast6MonthsCA.transpose()



totalSalesPerDayPast6MonthsTX = salesTrainValidationData.copy().groupby('store_id').sum().loc["TX_1":"TX_3", "d_1733":"d_1913"].transpose()

totalSalesPerDayPast6MonthsTX = totalSalesPerDayPast6MonthsTX.set_index(pd.to_datetime(dates))

totalSalesPerDayPast6MonthsTX.transpose()



totalSalesPerDayPast6MonthsWI = salesTrainValidationData.copy().groupby('store_id').sum().loc["WI_1":"WI_3", "d_1733":"d_1913"].transpose()

totalSalesPerDayPast6MonthsWI = totalSalesPerDayPast6MonthsWI.set_index(pd.to_datetime(dates))

totalSalesPerDayPast6MonthsWI.transpose()



f,ax = plt.subplots(3,1,figsize=(20,8))

f.tight_layout(pad=3.0)

plt.style.use('bmh')



ax[0].plot(totalSalesPerDayPast6MonthsCA)

ax[0].set_title('Sales by store in CA')

ax[0].legend(('CA_1', 'CA_2', 'CA_3', 'CA_4'), loc='upper left')



ax[1].plot(totalSalesPerDayPast6MonthsTX)

ax[1].set_title('Sales by store in TX')

ax[1].legend(('TX_1', 'TX_2', 'TX_3'), loc='upper left')



ax[2].plot(totalSalesPerDayPast6MonthsWI)

ax[2].set_title('Sales by store in WI')

ax[2].legend(('WI_1', 'WI_2', 'WI_3'), loc='upper left')

plt.show()
totalSalesPerDayPast6Months = salesTrainValidationData.copy().groupby('cat_id').sum().loc[:,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

totalSalesPerDayPast6Months = totalSalesPerDayPast6Months.set_index(pd.to_datetime(dates))



totalSalesPerDayPast6Months.plot(title="Total sales by categories across all stores", figsize=(30,8))

plt.style.use('bmh')

plt.show()
totalSalesPerDayPast6Months = salesTrainValidationData.copy().groupby(by=['state_id','cat_id'], axis=0).sum().loc[:,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

totalSalesPerDayPast6Months = totalSalesPerDayPast6Months.set_index(pd.to_datetime(dates))



f,ax = plt.subplots(3,1,figsize=(20,8))

f.tight_layout(pad=3.0)

plt.style.use('bmh')



ax[0].plot(totalSalesPerDayPast6Months['CA'])

ax[0].set_title('Sales by category in CA')

ax[0].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')



ax[1].plot(totalSalesPerDayPast6Months['TX'])

ax[1].set_title('Sales by category in TX')

ax[1].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')



ax[2].plot(totalSalesPerDayPast6Months['WI'])

ax[2].set_title('Sales by category in WI')

ax[2].legend(('FOODS', 'HOBBIES', 'HOUSEHOLD'), loc='upper left')

plt.show()
totalSalesPerDayPast6Months = salesTrainValidationData.copy().groupby('dept_id').sum().loc[:,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

totalSalesPerDayPast6Months = totalSalesPerDayPast6Months.set_index(pd.to_datetime(dates))



totalSalesPerDayPast6Months.plot(title="Total sales by subcategories across all stores", figsize=(30,8))

plt.style.use('bmh')

plt.legend(loc='upper left')

plt.show()
caSales = salesTrainValidationData.copy().loc[salesTrainValidationData['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]

caSales = caSales.groupby('dept_id').sum().loc[:,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

caSales = caSales.set_index(pd.to_datetime(dates))



txSales = salesTrainValidationData.copy().loc[salesTrainValidationData['store_id'].isin(['TX_1', 'TX_2', 'TX_3'])]

txSales = txSales.groupby('dept_id').sum().loc[:,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

txSales = txSales.set_index(pd.to_datetime(dates))



wiSales = salesTrainValidationData.copy().loc[salesTrainValidationData['store_id'].isin(['WI_1', 'WI_2', 'WI_3'])]

wiSales = wiSales.groupby('dept_id').sum().loc[:,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

wiSales = wiSales.set_index(pd.to_datetime(dates))



f,ax = plt.subplots(3,1,figsize=(24,8))

f.tight_layout(pad=3.0)

plt.style.use('bmh')



ax[0].plot(caSales)

ax[0].set_title('Total sales of subcategories for CA')

ax[0].legend(('FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2'), loc='upper left')

ax[0].set_ylabel('Sales')



ax[1].plot(txSales)

ax[1].set_title('Total sales of subcategories for TX')

ax[1].legend(('FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2'), loc='upper left')

ax[1].set_ylabel('Sales')



ax[2].plot(wiSales)

ax[2].set_title('Total sales of subcategories for WI')

ax[2].legend(('FOODS_1', 'FOODS_2', 'FOODS_3', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2'), loc='upper left')

ax[2].set_ylabel('Sales')



plt.show()
foods = salesTrainValidationData.copy().loc[salesTrainValidationData['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]

foods = foods.groupby('dept_id').sum().loc["FOODS_1":"FOODS_3":,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

foods = foods.set_index(pd.to_datetime(dates))



hobbies = salesTrainValidationData.copy().loc[salesTrainValidationData['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]

hobbies = hobbies.groupby('dept_id').sum().loc["HOBBIES_1":"HOBBIES_2":,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

hobbies = hobbies.set_index(pd.to_datetime(dates))



household = salesTrainValidationData.copy().loc[salesTrainValidationData['store_id'].isin(['CA_1', 'CA_2', 'CA_3', 'CA_4'])]

household = household.groupby('dept_id').sum().loc["HOUSEHOLD_1":"HOUSEHOLD_2":,"d_1733":"d_1913"].transpose()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

household = household.set_index(pd.to_datetime(dates))



f,ax = plt.subplots(3,1,figsize=(24,8))

f.tight_layout(pad=3.0)

plt.style.use('bmh')



ax[0].plot(foods)

ax[0].set_title('Overall sales for food subcategories in CA with snap days')

ax[0].legend(('FOODS_1', 'FOODS_2', 'FOODS_3'), loc='upper left')



for date in getSnapDaysInRange(1733, 1913, 'CA'):

    ax[0].axvline(pd.Timestamp(date), color='#cfb4b4', zorder=1, linewidth=3)



ax[1].plot(hobbies)

ax[1].set_title('Overall sales for hobby subcategories in CA with snap days')

ax[1].legend(('HOBBIES_1', 'HOBBIES_2'), loc='upper left')



for date in getSnapDaysInRange(1733, 1913, 'CA'):

    ax[1].axvline(pd.Timestamp(date), color='#cfb4b4', zorder=1, linewidth=3)



ax[2].plot(household)

ax[2].set_title('Overall sales for household subcategories in CA with snap days')

ax[2].legend(('HOUSEHOLD_1', 'HOUSEHOLD_2'), loc='upper left')



for date in getSnapDaysInRange(1733, 1913, 'CA'):

    ax[2].axvline(pd.Timestamp(date), color='#cfb4b4', zorder=1, linewidth=3)



plt.show()
caSales = salesTrainValidationData.copy().groupby('store_id').sum().loc['CA_1':'CA_4',"d_1733":"d_1913"].sum().transpose().to_frame()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

caSales = caSales.set_index(pd.to_datetime(dates))



txSales = salesTrainValidationData.copy().groupby('store_id').sum().loc['TX_1':'TX_3',"d_1733":"d_1913"].sum().transpose().to_frame()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

txSales = txSales.set_index(pd.to_datetime(dates))



wiSales = salesTrainValidationData.copy().groupby('store_id').sum().loc['WI_1':'WI_3',"d_1733":"d_1913"].sum().transpose().to_frame()

dates = pd.to_datetime(calendarData.copy().loc[1733:1913].loc[:,'date'].values)

wiSales = wiSales.set_index(pd.to_datetime(dates))



f,ax = plt.subplots(3,1,figsize=(24,8))

f.tight_layout(pad=3.0)

plt.style.use('bmh')



ax[0].plot(caSales, zorder=2)

ax[0].set_title('Total sales for CA with snap days')



for date in getSnapDaysInRange(1733, 1913, 'CA'):

    ax[0].axvline(pd.Timestamp(date), color='#cfb4b4', zorder=1, linewidth=3)



ax[1].plot(txSales, zorder=2)

ax[1].set_title('Total sales for TX with snap days')



for date in getSnapDaysInRange(1733, 1913, 'TX'):

    ax[1].axvline(pd.Timestamp(date), color='#cfb4b4', zorder=1, linewidth=3)



ax[2].plot(wiSales, zorder=2)

ax[2].set_title('Total sales for WI with snap days')



for date in getSnapDaysInRange(1733, 1913, 'WI'):

    ax[2].axvline(pd.Timestamp(date), color='#cfb4b4', zorder=1, linewidth=3)



plt.show()
def calculateSlope(series, n, sumX):

    def a():

        a = 0

        for i in range (0, n):

            a += (i + 1) * series.iloc[i]

        return a * n

    def b():

        sumY = 0

        for i in range (0, n):

            sumY += i + 1

        return series.sum() * sumY

    def c():

        c = 0

        for i in range (0, n):

            c += (i + 1)**2

        return c * n

    def d():

        return sumX**2

    return (a() - b()) / (c() - d())



def calculateYIntercept(series, n, m, sumX):

    def e():

        return series.sum()

    def f():

        return m * sumX

    return (e() - f()) / n



def calculateTrend(series, count):

    n = len(series.index)



    sumX = 0

    for i in range (0, n):

        sumX += i + 1



    m = calculateSlope(series, n, sumX)

    yi = calculateYIntercept(series, n, m, sumX)



    trend = []

    for i in range(n + 1, n + 1 + count):

        trend.append(m * i + yi)



    return pd.Series(trend)
train = salesTrainValidationData.copy().loc[:,'d_1':'d_1885'].sum()

validation = salesTrainValidationData.copy().loc[:,'d_1886':'d_1913'].sum()

trend = calculateTrend(train, 28)

predicted = trend.to_frame().set_index(validation.index.values).transpose().iloc[0]

rmse = math.sqrt(metrics.mean_squared_error(validation, predicted))



compare = pd.DataFrame({

    'Actual': validation,

    'Predicted': predicted

})



dates = pd.to_datetime(calendarData.copy().loc[1886:1913].loc[:,'date'].values)

compare = compare.set_index(pd.to_datetime(dates))

compare

compare.plot(title='Actual total sales with trendline prediction RMSE (' + str(rmse) + ')', figsize=(24,4))

plt.style.use('bmh')

plt.xlabel('')

plt.ylabel('Sales')

plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX



validation = salesTrainValidationData.copy().loc[:,'d_1886':'d_1913'].sum()

endog = salesTrainValidationData.copy().loc[:,'d_1':'d_1885'].sum().reset_index().drop(columns=['index'])

model = SARIMAX(endog, order=(4, 1, 1), seasonal_order=(4, 1, 1, 7))

fit = model.fit()

yhat = fit.forecast(28)

rmse = math.sqrt(metrics.mean_squared_error(validation, yhat.values))



compare = pd.DataFrame({

    'Actual': validation,

    'Predicted': yhat.values

})



dates = pd.to_datetime(calendarData.copy().loc[1886:1913].loc[:,'date'].values)

compare = compare.set_index(pd.to_datetime(dates))

compare.plot(title='Actual total sales with SARIMAX prediction RMSE (' + str(rmse) + ')', figsize=(24,4))

plt.style.use('bmh')

plt.xlabel('')

plt.ylabel('Sales')

plt.show()
salesTrain = salesTrainValidationData.copy().loc[:,"d_1":"d_1885"].sum().to_frame()

salesValidation = salesTrainValidationData.copy().loc[:,"d_1886":"d_1913"].sum().to_frame()

salesTrain.columns = ['dailySales']

salesValidation.columns = ['dailySales']

datesTrain = calendarData[0:1885].copy()

datesValidation = calendarData[1885:1913].copy()

datesTrain = datesTrain[['weekday','month', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'wm_yr_wk', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']]

datesValidation = datesValidation[['weekday','month', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'wm_yr_wk', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']]

datesTrain['dailySales'] = salesTrain['dailySales'].values

datesValidation['dailySales'] = salesValidation['dailySales'].values

featuresTrain = pd.get_dummies(datesTrain)

featuresValidation = pd.get_dummies(datesValidation)



for c in featuresTrain.columns:

    if c not in featuresValidation.columns:

        featuresValidation[c] = c

        featuresValidation[c] = 0



labelsTrain = np.array(featuresTrain['dailySales'])

featuresTrain = featuresTrain.drop('dailySales', axis=1)

labelsValidation = np.array(featuresValidation['dailySales'])

featuresValidation = featuresValidation.drop('dailySales', axis=1)

featuresTrain = np.array(featuresTrain)

featuresValidation = np.array(featuresValidation)



rf = RandomForestRegressor(n_estimators=100, random_state=1)

rf.fit(featuresTrain, labelsTrain)

predictions = rf.predict(featuresValidation)

rmse = math.sqrt(metrics.mean_squared_error(labelsValidation, predictions))



compare = pd.DataFrame({

    'Actual': labelsValidation,

    'Predicted': predictions

})



dates = pd.to_datetime(calendarData.copy().loc[1886:1913].loc[:,'date'].values)

compare = compare.set_index(pd.to_datetime(dates))

compare

compare.plot(title='Actual total sales last 28 days of data with RMSE (' + str(rmse) + ')', figsize=(24,4))

plt.style.use('bmh')

plt.xlabel('')

plt.ylabel('Sales')

plt.show()
salesTrain = salesTrainValidationData.copy().loc[:,"d_1":"d_1285"].sum().to_frame()

salesValidation = salesTrainValidationData.copy().loc[:,"d_1286":"d_1913"].sum().to_frame()

salesTrain.columns = ['dailySales']

salesValidation.columns = ['dailySales']

datesTrain = calendarData[0:1285].copy()

datesValidation = calendarData[1285:1913].copy()

datesTrain = datesTrain[['weekday','month', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'wm_yr_wk', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']]

datesValidation = datesValidation[['weekday','month', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'wm_yr_wk', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']]

datesTrain['dailySales'] = salesTrain['dailySales'].values

datesValidation['dailySales'] = salesValidation['dailySales'].values

featuresTrain = pd.get_dummies(datesTrain)

featuresValidation = pd.get_dummies(datesValidation)



for c in featuresTrain.columns:

    if c not in featuresValidation.columns:

        featuresValidation[c] = c

        featuresValidation[c] = 0



labelsTrain = np.array(featuresTrain['dailySales'])

featuresTrain = featuresTrain.drop('dailySales', axis=1)

labelsValidation = np.array(featuresValidation['dailySales'])

featuresValidation = featuresValidation.drop('dailySales', axis=1)

featuresTrain = np.array(featuresTrain)

featuresValidation = np.array(featuresValidation)



rf = RandomForestRegressor(n_estimators=100, random_state=1)

rf.fit(featuresTrain, labelsTrain)

predictions = rf.predict(featuresValidation)

rmse = math.sqrt(metrics.mean_squared_error(labelsValidation, predictions))



compare = pd.DataFrame({

    'Actual': labelsValidation,

    'Predicted': predictions

})



dates = pd.to_datetime(calendarData.copy().loc[1286:1913].loc[:,'date'].values)

compare = compare.set_index(pd.to_datetime(dates))

compare

compare.plot(title='Actual total sales last 2 years of data with RMSE (' + str(rmse) + ')', figsize=(24,4))

plt.style.use('bmh')

plt.xlabel('')

plt.ylabel('Sales')

plt.show()