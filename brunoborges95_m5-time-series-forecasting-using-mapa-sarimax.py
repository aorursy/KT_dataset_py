import pandas as pd

import numpy as np

import plotly.graph_objs as go #visualization library

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #autocorrelation test

import statsmodels.api as sm

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller #stationarity test

from statsmodels.tsa.statespace.sarimax import SARIMAX 

from datetime import datetime, timedelta

import seaborn as sns

from sklearn import metrics

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')



data_dept = data.groupby(['dept_id']).sum() #group sales by department

data_item = data.groupby(['item_id']).sum() #group sales by item_id

data_cat = data.groupby(['cat_id']).sum().T #group sales by category

data_cat['day'] = data_cat.index



data_store = data.groupby(['store_id']).sum()

data_state_id = data.groupby(['state_id']).sum()



calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

data_calendar = calendar.iloc[:, [0, 2,3,4,5,6,7]]



#Merge data_calendar columns related to commemorative data, days of the week, month and year.

data_cat = pd.merge(data_calendar, data_cat, how = 'inner', left_on='d', right_on='day')

data_cat_final = data_cat.iloc[:,[7,8,9]]

data_cat_final.index = data_cat['date']

data_cat_final.index = pd.to_datetime(data_cat_final.index , format = '%Y-%m-%d')

data_cat_final.parse_dates=data_cat_final.index

data_cat_final.head(10)
fig = go.Figure(

    data=[go.Scatter(y=data_cat_final['2011-01':'2016-04'].FOODS, x=data_cat_final.index, name= 'Foods'), 

          go.Scatter(y=data_cat_final['2011-01':'2016-04'].HOBBIES, x=data_cat_final.index, name = 'Hobbies'),

          go.Scatter(y=data_cat_final['2011-01':'2016-04'].HOUSEHOLD, x=data_cat_final.index, name = 'HouseHold')],

    layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

    )

)

fig.update_layout(title_text="Sales by Category")

fig.show()
sns.heatmap(data_cat_final[['FOODS','HOBBIES','HOUSEHOLD']].corr(), annot = True,  cbar=False)
data_cat_final_monthly = data_cat_final.iloc[:,[0,1,2]].resample('M').sum()[2:-1] #mensal resampling

data_cat_final_weekly = data_cat_final.iloc[:,[0,1,2]].resample('W').sum()[8:-1] #weekly resampling

data_cat_final_bimonthly = data_cat_final.iloc[:,[0,1,2]].resample('2M').sum()[1:-1] #bimonthy resamply
fig = go.Figure(

    data=[go.Scatter(y=data_cat_final_monthly.FOODS, x=data_cat_final_monthly.FOODS.index, name= 'Foods'), 

          go.Scatter(y=data_cat_final_monthly.HOBBIES, x=data_cat_final_monthly.HOBBIES.index, name = 'Hobbies'),

          go.Scatter(y=data_cat_final_monthly.HOUSEHOLD, x=data_cat_final_monthly.HOUSEHOLD.index, name = 'HouseHold')],

    layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

    )

)

fig.update_layout(title_text="Sales by Category - Monthly")

fig.show()
decomposed = sm.tsa.seasonal_decompose(np.array(data_cat_final_monthly.FOODS),period=6) # The frequency is semestral

figure = decomposed.plot()

decomposed = sm.tsa.seasonal_decompose(np.array(data_cat_final_monthly.HOBBIES),period=6) # The frequency is semestral

figure = decomposed.plot()
#### Household Category - Decomposition
decomposed = sm.tsa.seasonal_decompose(np.array(data_cat_final_monthly.HOUSEHOLD),period=6) # The frequency is semestral

figure = decomposed.plot()
plt.show()

plot_acf(data_cat_final_monthly.FOODS,lags=12,title="ACF Foods")

plt.show()

plot_pacf(data_cat_final_monthly.FOODS,lags=6,title="PACF Foods")

plt.show()
plot_acf(data_cat_final_monthly.HOBBIES,lags=12,title="ACF HObbies")

plt.show()

plot_pacf(data_cat_final_monthly.HOBBIES,lags=12,title="PACF Hobbies")

plt.show()
plot_acf(data_cat_final_monthly.HOUSEHOLD,lags=12,title="ACF Household")

plt.show()

plot_pacf(data_cat_final_monthly.HOUSEHOLD,lags=12,title="PACF Household")

plt.show()
# Augmented Dickey-Fuller test

adf1 = adfuller(data_cat_final.FOODS, autolag='AIC')

print("p-value of Foods serie is: {}".format(float(adf1[1])))

#The test statistic is negative, we don't reject the null hypothesis (it looks non-stationary).

adf2 = adfuller(data_cat_final.HOBBIES, autolag='AIC')

print("p-value of Hobbies serie is: {}".format(float(adf2[1]))) #It isn't a random walk if p-value is less than 5%

#The test statistic is negative, we don't reject the null hypothesis (it looks non-stationary).

adf3 = adfuller(data_cat_final.HOUSEHOLD, autolag='AIC')

print("p-value of Household serie is: {}".format(float(adf3[1]))) #It isn't a random walk if p-value is less than 5%

#The test statistic is negative, we don't reject the null hypothesis (it looks non-stationary).
#I created a function that should return a forecast and a summary with the main statistics.

#actual - Time series that we will predict

#order - [p, d, q] terms

#seasonal order - [P,S,Q,m] sazonal terms

#t- lag use for the test base and future prediction

def sarimax_predictor(actual, order, seasonal_order, t , start, title):

    mdl = sm.tsa.statespace.SARIMAX(actual[start:-t],

                                            order=order, seasonal_order=seasonal_order,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

    results = mdl.fit()

    results.plot_diagnostics()

    print(results.summary())

    predict = results.predict(start=start, end=len(actual)+t)



    fig = go.Figure(

        data=[go.Scatter(y=actual[0:-t],x=actual[0:-t].index, name= 'Actual'),

          go.Scatter(y=actual[-t-1::],x=actual[-t-1::].index, name= 'Test'),

          go.Scatter(y=predict, x=predict.index, name= 'Predict')],

        layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

        )

    )

    fig.update_layout(title_text= title)

    fig.show()

    return predict

#defined function when we need to use exogenous variables

def sarimax_predictor_exog(actual, order, seasonal_order, t, title, exog):

    mdl = sm.tsa.statespace.SARIMAX(actual[0:-t],

                                            order=order, seasonal_order=seasonal_order, exog = exog[0:-t],

                                            enforce_stationarity=False,

                                            enforce_invertibility=False, time_varying_regression = False,

                                            mle_regression = True)

    results = mdl.fit()

    results.plot_diagnostics()

    print(results.summary())

    #use only exogenous to forecasting (test set)

    predict = results.predict(start=0, end=len(actual), exog=exog[-t-1::]) 



    fig = go.Figure(

        data=[go.Scatter(y=actual[0:-t],x=actual[0:-t].index, name= 'Actual'),

          go.Scatter(y=actual[-t-1::],x=actual[-t-1::].index, name= 'Test'),

          go.Scatter(y=predict, x=predict.index, name= 'Predict')],

        layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

        )

    )

    fig.update_layout(title_text= title)

    fig.show()

    return predict



#evaluation test

def rmse(actual, predict, title):

    from sklearn import metrics

    rmse = np.sqrt(metrics.mean_squared_error(actual, predict))

    print('The RMSE of ' + title + ' is:', rmse)
start = 0

predicted_result_foods_weekly = sarimax_predictor(data_cat_final_weekly.FOODS, [1,1,0], [1,1,0,24], 7*4, start,

                                                  'Weekly forecast - Foods')
predicted_result_foods_monthly = sarimax_predictor(data_cat_final_monthly.FOODS, [5,1,1], [1,1,0,6], 6, start,

                                                  'Monthly forecast- Foods')
predicted_result_foods_bimonthly = sarimax_predictor(data_cat_final_bimonthly.FOODS, [2,1,0], [1,0,0,6], 3, start,

                                                     'Bimonthly forecast')
#RMSE in test fold

rmse_foods_weekly= rmse(data_cat_final_weekly.FOODS[-28::], predicted_result_foods_weekly[-56-1:-28-1], 'weekly Foods - Test')

rmse_foods_monthly= rmse(data_cat_final_monthly.FOODS[-6::], predicted_result_foods_monthly[-12-1:-6-1], 'monthy Foods - Test')

rmse_foods_bimonthly= rmse(data_cat_final_bimonthly.FOODS[-3::], predicted_result_foods_bimonthly[-6-1:-3-1], 'bimonthy Foods - Test')



#RMSE in train fold

rmse_foods_weekly= rmse(data_cat_final_weekly.FOODS[start:-28], predicted_result_foods_weekly[0:-28*2-1], 'weekly Foods - Train')

rmse_foods_monthly= rmse(data_cat_final_monthly.FOODS[start:-6], predicted_result_foods_monthly[0:-6*2-1], 'monthy Foods - Train')

rmse_foods_bimonthly= rmse(data_cat_final_bimonthly.FOODS[start:-3], predicted_result_foods_bimonthly[0:-3*2-1], 'bimonthy Foods - Train')
predicted_result_hobbies_weekly = sarimax_predictor(data_cat_final_weekly.HOBBIES, [2,1,0], [1,1,0,24], 28, start,

                                                  'Weekly forecast - Hobbies')
predicted_result_hobbies_monthly = sarimax_predictor(data_cat_final_monthly.HOBBIES, [2,1,0], [2,0,0,12], 6, start,

                                                  'Monthly forecast- Hobbies')
predicted_result_hobbies_bimonthly = sarimax_predictor(data_cat_final_bimonthly.HOBBIES, [2,1,0], [1,0,0,3], 3, start,

                                                     'Bimonthly forecast - Hobbies')
#RMSE in test fold

rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[-28::], predicted_result_hobbies_weekly[-56-1:-28-1], 'weekly Hobbies - Test')

rmse_hobbies_monthly= rmse(data_cat_final_monthly.HOBBIES[-6::], predicted_result_hobbies_monthly[-12-1:-6-1], 'monthy Hobbies - Test')

rmse_hobbies_bimonthly= rmse(data_cat_final_bimonthly.HOBBIES[-3::], predicted_result_hobbies_bimonthly[-6-1:-3-1], 'bimonthy Hobbies - Test')



#RMSE in train fold

rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[start:-28], predicted_result_hobbies_weekly[0:-28*2-1], 'weekly Hobbies - Train')

rmse_hobbies_monthly= rmse(data_cat_final_monthly.HOBBIES[start:-6], predicted_result_hobbies_monthly[0:-6*2-1], 'monthy Hobbies - Train')

rmse_hobbies_bimonthly= rmse(data_cat_final_bimonthly.HOBBIES[start:-3], predicted_result_hobbies_bimonthly[0:-3*2-1], 'bimonthy Hobbies - Train')
predicted_result_household_weekly = sarimax_predictor(data_cat_final_weekly.HOUSEHOLD, [7,1,0], [1,1,0,24], 28, start,

                                                  'Weekly forecast - Household')
predicted_result_household_monthly = sarimax_predictor(data_cat_final_monthly.HOUSEHOLD, [2,1,0], [2,0,0,6], 6, start,

                                                  'Monthly forecast- Household')
predicted_result_household_bimonthly = sarimax_predictor(data_cat_final_bimonthly.HOUSEHOLD, [2,0,0], [1,1,0,3], 3, start,

                                                     'Bimonthly forecast - Household')
#RMSE in test fold

rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[-28::], predicted_result_household_weekly[-56-1:-28-1], 'weekly Household - Test')

rmse_household_monthly= rmse(data_cat_final_monthly.HOUSEHOLD[-6::], predicted_result_household_monthly[-12-1:-6-1], 'monthy Household - Test')

rmse_household_bimonthly= rmse(data_cat_final_bimonthly.HOUSEHOLD[-3::], predicted_result_household_bimonthly[-6-1:-3-1], 'bimonthy Household - Test')



#RMSE in train fold

rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[start:-28], predicted_result_household_weekly[0:-28*2-1], 'weekly Household - Train')

rmse_household_monthly= rmse(data_cat_final_monthly.HOUSEHOLD[start:-6], predicted_result_household_monthly[0:-6*2-1], 'monthy Household - Train')

rmse_household_bimonthly= rmse(data_cat_final_bimonthly.HOUSEHOLD[start:-3], predicted_result_household_bimonthly[0:-3*2-1], 'bimonthy Household - Train')
def alpha(actual, predict):

    RMSE =[]

    for i in np.arange(0.5,15, 0.01):

        RMSE.append([i,np.sqrt(metrics.mean_squared_error(actual, predict*i))])

    return np.array(RMSE)[np.argmin(np.array(RMSE)[:,1]),0]
#step of combination

predicted_result_foods_bimonthly = predicted_result_foods_bimonthly.resample('W').mean()

predicted_result_foods_monthly = predicted_result_foods_monthly.resample('W').mean()

#equally assigns the mean value of the low frequency time series

predictions_foods = pd.DataFrame({'bimonthly': predicted_result_foods_bimonthly.groupby(predicted_result_foods_bimonthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))[0:-4],

                         'monthly': predicted_result_foods_monthly.groupby(predicted_result_foods_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x)),

                         'weekly': predicted_result_foods_weekly[1::]})

prediction_foods_mean = pd.DataFrame.mean(predictions_foods, axis = 1)

prediction_foods_median = pd.DataFrame.median(predictions_foods, axis = 1)

prediction_foods_min = pd.DataFrame.min(predictions_foods, axis = 1)

prediction_foods_max = pd.DataFrame.max(predictions_foods, axis = 1)

alpha_foods_max = alpha(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_max[start:-28*2-1])

alpha_foods_min = alpha(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_min[start:-28*2-1])

alpha_foods_mean = alpha(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_mean[start:-28*2-1])

alpha_median = alpha(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_median[start:-28*2-1])



fig = go.Figure(

    data=[go.Scatter(y=data_cat_final_weekly.FOODS, x = data_cat_final_weekly.FOODS.index, name= 'Actual'), 

          go.Scatter(y=prediction_foods_min[0:-1]*alpha_foods_min, x= data_cat_final_weekly.FOODS.index, name= 'Predict Min'),

          go.Scatter(y=prediction_foods_max[0:-1]*alpha_foods_max, x= data_cat_final_weekly.FOODS.index, name= 'Predict Max'),

          go.Scatter(y=prediction_foods_mean[0:-1]*alpha_foods_mean, x= data_cat_final_weekly.FOODS.index, name= 'Predict Mean'),

          go.Scatter(y=prediction_foods_median[0:-1]*alpha_median, x= data_cat_final_weekly.FOODS.index, name= 'Predict median')],

    layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

    )

)

fig.update_layout(title_text="Foods Category - MAPA SARIMA forecast")

fig.show()
rmse(data_cat_final_weekly.FOODS[-28::], predicted_result_foods_weekly[-56-1:-28-1], 'weekly Foods - Test')

rmse(data_cat_final_weekly.FOODS[start+1:-28], predicted_result_foods_weekly[start+1:-28*2-1], 'weekly Foods - Train')

#test Fold

rmse_foods_max = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_max[-57:-29]*alpha_foods_max, 'Foods Test - max')

rmse_foods_min = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_min[-57:-29]*alpha_foods_min, 'Foods Test - min')

rmse_foods_mean = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_mean[-57:-29]*alpha_foods_mean, 'Foods Test - mean')

rmse_foods_median = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_median[-57:-29]*alpha_median, 'Foods Test - median')

#train Fold

rmse_foods_max = rmse(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_max[start:-28*2-1]*alpha_foods_max, 'Foods Train - max')

rmse_foods_min = rmse(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_min[start:-28*2-1]*alpha_foods_min, 'Foods Train - min')

rmse_foods_mean = rmse(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_mean[start:-28*2-1]*alpha_foods_mean, 'Foods Train - mean')

rmse_foods_median = rmse(data_cat_final_weekly.FOODS[start+1:-28], prediction_foods_median[start:-28*2-1]*alpha_median, 'Foods Train - median' )
predicted_result_hobbies_bimonthly = predicted_result_hobbies_bimonthly.resample('W').mean()

predicted_result_hobbies_monthly = predicted_result_hobbies_monthly.resample('W').mean()

predictions_hobbies = pd.DataFrame({'bimestral': predicted_result_hobbies_bimonthly.groupby(predicted_result_hobbies_bimonthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))[0:-4],

                         'mensal': predicted_result_hobbies_monthly.groupby(predicted_result_hobbies_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x)),

                         'weekly': predicted_result_hobbies_weekly[1::]})

prediction_hobbies_mean = pd.DataFrame.mean(predictions_hobbies, axis = 1)

prediction_hobbies_median = pd.DataFrame.median(predictions_hobbies, axis = 1)

prediction_hobbies_min = pd.DataFrame.min(predictions_hobbies, axis = 1)

prediction_hobbies_max = pd.DataFrame.max(predictions_hobbies, axis = 1)



alpha_hobbies_max = alpha(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_max[start:-28*2-1])

alpha_hobbies_min = alpha(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_min[start:-28*2-1])

alpha_hobbies_mean = alpha(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_mean[start:-28*2-1])

alpha_hobbies_median = alpha(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_median[start:-28*2-1])



fig = go.Figure(

    data=[go.Scatter(y=data_cat_final_weekly.HOBBIES, x = data_cat_final_weekly.HOBBIES.index, name= 'Actual'), 

          go.Scatter(y=prediction_hobbies_min[0:-1]*alpha_hobbies_min, x = data_cat_final_weekly.HOBBIES.index,name= 'Predict Min'),

          go.Scatter(y=prediction_hobbies_max[0:-1]*alpha_hobbies_max, x = data_cat_final_weekly.HOBBIES.index, name= 'Predict Max'),

          go.Scatter(y=prediction_hobbies_mean[0:-1]*alpha_hobbies_mean, x = data_cat_final_weekly.HOBBIES.index, name= 'Predict Mean'),

          go.Scatter(y=prediction_hobbies_median[0:-1]*alpha_hobbies_median, x = data_cat_final_weekly.HOBBIES.index, name= 'Predict median')],

    layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

    )

)

fig.update_layout(title_text="Hobbies Category - MAPA SARIMA forecast")

fig.show()
rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[-28::], predicted_result_hobbies_weekly[-56-1:-28-1], 'weekly Hobbies - Test')

rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[start+1:-28], predicted_result_hobbies_weekly[start+1:-28*2-1], 'weekly Hobbies - Train')



#test Fold

rmse_hobbies_max = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_max[-57:-29]*alpha_hobbies_max, 'Hobbies Test - max')

rmse_hobbies_min = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_min[-57:-29]*alpha_hobbies_min, 'Hobbies Test - min')

rmse_hobbies_mean = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_mean[-57:-29]*alpha_hobbies_mean, 'Hobbies Test - mean')

rmse_hobbies_median = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_median[-57:-29]*alpha_hobbies_median, 'Hobbies Test - median')

#train Fold

rmse_hobbies_max = rmse(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_max[start:-28*2-1]*alpha_hobbies_max, 'Hobbies Train - max')

rmse_hobbies_min = rmse(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_min[start:-28*2-1]*alpha_hobbies_min, 'Hobbies Train - min')

rmse_hobbies_mean = rmse(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_mean[start:-28*2-1]*alpha_hobbies_mean, 'Hobbies Train - mean')

rmse_hobbies_median = rmse(data_cat_final_weekly.HOBBIES[start+1:-28], prediction_hobbies_median[start:-28*2-1]*alpha_hobbies_median, 'Hobbies Train - median' )
predicted_result_household_bimonthly = predicted_result_household_bimonthly.resample('W').mean()

predicted_result_household_monthly = predicted_result_household_monthly.resample('W').mean()

predictions_household = pd.DataFrame({'bimestral': predicted_result_household_bimonthly.groupby(predicted_result_household_bimonthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))[0:-4],

                         'mensal': predicted_result_household_monthly.groupby(predicted_result_household_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x)),

                         'weekly': predicted_result_household_weekly[1::]})

prediction_household_mean = pd.DataFrame.mean(predictions_household, axis = 1)

prediction_household_median = pd.DataFrame.median(predictions_household, axis = 1)

prediction_household_min = pd.DataFrame.min(predictions_household, axis = 1)

prediction_household_max = pd.DataFrame.max(predictions_household, axis = 1)



alpha_household_max = alpha(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_max[start:-28*2-1])

alpha_household_min = alpha(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_min[start:-28*2-1])

alpha_household_mean = alpha(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_mean[start:-28*2-1])

alpha_household_median = alpha(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_median[start:-28*2-1])

fig = go.Figure(

    data=[go.Scatter(y=data_cat_final_weekly.HOUSEHOLD, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Actual'), 

          go.Scatter(y=prediction_household_min[0:-1]*alpha_household_min, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Min'),

          go.Scatter(y=prediction_household_max[0:-1]*alpha_household_max, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Max'),

          go.Scatter(y=prediction_household_mean[0:-1]*alpha_household_mean, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Mean'),

          go.Scatter(y=prediction_household_median[0:-1]*alpha_household_median, x=data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict median')],

    layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

    )

)

fig.update_layout(title_text="Household Category - MAPA SARIMA forecast")

fig.show()
rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[-28::], predicted_result_household_weekly[-56-1:-28-1], 'weekly Household - Test')

rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[start:-28], predicted_result_household_weekly[start:-28*2-1], 'weekly Household - Train')



#test Fold

rmse_household_max = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_max[-57:-29]*alpha_household_max, 'Household Test - max')

rmse_household_min = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_min[-57:-29]*alpha_household_min, 'Household Test - min')

rmse_household_mean = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_mean[-57:-29]*alpha_household_mean, 'Household Test - mean')

rmse_household_median = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_median[-57:-29]*alpha_household_median, 'Household Test - median')

#train Fold

rmse_household_max = rmse(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_max[start:-28*2-1]*alpha_household_max, 'Household Train - max')

rmse_household_min = rmse(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_min[start:-28*2-1]*alpha_household_min, 'Household Train - min')

rmse_household_mean = rmse(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_mean[start:-28*2-1]*alpha_household_mean, 'Household Train - mean')

rmse_household_median = rmse(data_cat_final_weekly.HOUSEHOLD[start+1:-28], prediction_household_median[start:-28*2-1]*alpha_household_median, 'Household Train - median' )
holidays = pd.get_dummies(data_cat['event_name_1'], dummy_na=True)

weekdays = pd.get_dummies(data_cat['wday'])

exog = pd.concat([holidays, weekdays], axis = 1)

exog.index = pd.to_datetime(data_cat_final.index , format = '%Y-%m-%d')

exog.head(10)
predicted_result_foods_daily = sarimax_predictor_exog(data_cat_final.FOODS, [2,1,0], [2,1,0,7], 28*7,

                        'Daily forecast - Foods', exog)
predicted_result_hobbies_daily = sarimax_predictor_exog(data_cat_final.HOBBIES, [6,1,1], [1,1,0,7], 28*7,

                        'Daily forecast - Hobbies', exog)
predicted_result_household_daily = sarimax_predictor_exog(data_cat_final.HOUSEHOLD, [3,0,0], [1,1,0,7], 28*7,

                        'Daily forecast - Household', exog)
predicted_result_foods_monthly = predicted_result_foods_monthly.resample('W').mean()

predictions_foods = pd.DataFrame({'mensal': predicted_result_foods_monthly.groupby(predicted_result_foods_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))['2011-04-03':'2016-05-01'],

                         'weekly': predicted_result_foods_weekly[1::]['2011-04-03':'2016-05-01'],

                          'daily': predicted_result_hobbies_daily.resample('W').sum()['2011-04-03':'2016-05-01']})

prediction_foods_mean = pd.DataFrame.mean(predictions_foods, axis = 1)

prediction_foods_median = pd.DataFrame.median(predictions_foods, axis = 1)

prediction_foods_min = pd.DataFrame.min(predictions_foods, axis = 1)

prediction_foods_max = pd.DataFrame.max(predictions_foods, axis = 1)

alpha_foods_max = alpha(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_max['2011-04-03':'2015-03-29 '])

alpha_foods_min = alpha(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_min['2011-04-03':'2015-03-29 '])

alpha_foods_mean = alpha(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_mean['2011-04-03':'2015-03-29 '])

alpha_median = alpha(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_median['2011-04-03':'2015-03-29 '])



fig = go.Figure(

    data=[go.Scatter(y=data_cat_final_weekly.FOODS, x= data_cat_final_weekly.FOODS.index, name= 'Actual'), 

          go.Scatter(y=prediction_foods_min[0:-1]*alpha_foods_min, x= data_cat_final_weekly.FOODS.index, name= 'Predict Min'),

          go.Scatter(y=prediction_foods_max[0:-1]*alpha_foods_max, x= data_cat_final_weekly.FOODS.index, name= 'Predict Max'),

          go.Scatter(y=prediction_foods_mean[0:-1]*alpha_foods_mean, x= data_cat_final_weekly.FOODS.index, name= 'Predict Mean'),

          go.Scatter(y=prediction_foods_median[0:-1]*alpha_median, x= data_cat_final_weekly.FOODS.index, name= 'Predict median')],

    layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

    )

)

fig.update_layout(title_text="Foods Category - MAPA SARIMA forecast")

fig.show()
rmse(data_cat_final_weekly.FOODS[-28::], predicted_result_foods_weekly[-56-1:-28-1], 'weekly Foods - Test')

rmse(data_cat_final_weekly.FOODS[start+1:-28], predicted_result_foods_weekly[start+1:-28*2-1], 'weekly Foods - Train')

#test Fold

rmse_foods_max = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_max[-57:-29]*alpha_foods_max, 'Foods Test - max')

rmse_foods_min = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_min[-57:-29]*alpha_foods_min, 'Foods Test - min')

rmse_foods_mean = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_mean[-57:-29]*alpha_foods_mean, 'Foods Test - mean')

rmse_foods_median = rmse(data_cat_final_weekly.FOODS[-28::], prediction_foods_median[-57:-29]*alpha_median, 'Foods Test - median')

#train Fold

rmse_foods_max = rmse(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_max['2011-04-03':'2015-03-29 ']*alpha_foods_max, 'Foods Train - max')

rmse_foods_min = rmse(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_min['2011-04-03':'2015-03-29 ']*alpha_foods_min, 'Foods Train - min')

rmse_foods_mean = rmse(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_mean['2011-04-03':'2015-03-29 ']*alpha_foods_mean, 'Foods Train - mean')

rmse_foods_median = rmse(data_cat_final_weekly.FOODS['2011-04-03':'2015-03-29 '], prediction_foods_median['2011-04-03':'2015-03-29 ']*alpha_median, 'Foods Train - median' )
predicted_result_hobbies_monthly = predicted_result_hobbies_monthly.resample('W').mean()

predictions_hobbies = pd.DataFrame({'mensal': predicted_result_hobbies_monthly.groupby(predicted_result_hobbies_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))['2011-04-03':'2016-05-01'],

                         'weekly': predicted_result_hobbies_weekly['2011-04-03':'2016-05-01'],

                         'daily': predicted_result_hobbies_daily.resample('W').sum()['2011-04-03':'2016-05-01']})

prediction_hobbies_mean = pd.DataFrame.mean(predictions_hobbies, axis = 1)

prediction_hobbies_median = pd.DataFrame.median(predictions_hobbies, axis = 1)

prediction_hobbies_min = pd.DataFrame.min(predictions_hobbies, axis = 1)

prediction_hobbies_max = pd.DataFrame.max(predictions_hobbies, axis = 1)



alpha_hobbies_max = alpha(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_max['2011-04-03':'2015-03-29 '])

alpha_hobbies_min = alpha(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_min['2011-04-03':'2015-03-29 '])

alpha_hobbies_mean = alpha(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_mean['2011-04-03':'2015-03-29 '])

alpha_hobbies_median = alpha(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_median['2011-04-03':'2015-03-29 '])



fig = go.Figure(

    data=[go.Scatter(y=data_cat_final_weekly.HOBBIES, x= data_cat_final_weekly.HOBBIES.index, name= 'Actual'), 

          go.Scatter(y=prediction_hobbies_min[0:-1]*alpha_hobbies_min, x= data_cat_final_weekly.HOBBIES.index, name= 'Predict Min'),

          go.Scatter(y=prediction_hobbies_max[0:-1]*alpha_hobbies_max, x= data_cat_final_weekly.HOBBIES.index, name= 'Predict Max'),

          go.Scatter(y=prediction_hobbies_mean[0:-1]*alpha_hobbies_mean, x= data_cat_final_weekly.HOBBIES.index, name= 'Predict Mean'),

          go.Scatter(y=prediction_hobbies_median[0:-1]*alpha_hobbies_median, x= data_cat_final_weekly.HOBBIES.index, name= 'Predict median')],

    layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

    )

)

fig.update_layout(title_text="Hobbies Category - MAPA SARIMA forecast")

fig.show()
rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[-28::], predicted_result_hobbies_weekly[-56-1:-28-1], 'weekly Hobbies - Test')

rmse_hobbies_weekly= rmse(data_cat_final_weekly.HOBBIES[start+1:-28], predicted_result_hobbies_weekly[start+1:-28*2-1], 'weekly Hobbies - Train')



#test Fold

rmse_hobbies_max = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_max[-57:-29]*alpha_hobbies_max, 'Hobbies Test - max')

rmse_hobbies_min = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_min[-57:-29]*alpha_hobbies_min, 'Hobbies Test - min')

rmse_hobbies_mean = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_mean[-57:-29]*alpha_hobbies_mean, 'Hobbies Test - mean')

rmse_hobbies_median = rmse(data_cat_final_weekly.HOBBIES[-28::], prediction_hobbies_median[-57:-29]*alpha_hobbies_median, 'Hobbies Test - median')

#train Fold

rmse_hobbies_max = rmse(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_max['2011-04-03':'2015-03-29 ']*alpha_hobbies_max, 'Hobbies Train - max')

rmse_hobbies_min = rmse(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_min['2011-04-03':'2015-03-29 ']*alpha_hobbies_min, 'Hobbies Train - min')

rmse_hobbies_mean = rmse(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_mean['2011-04-03':'2015-03-29 ']*alpha_hobbies_mean, 'Hobbies Train - mean')

rmse_hobbies_median = rmse(data_cat_final_weekly.HOBBIES['2011-04-03':'2015-03-29 '], prediction_hobbies_median['2011-04-03':'2015-03-29 ']*alpha_hobbies_median, 'Hobbies Train - median' )
predicted_result_household_bimonthly = predicted_result_household_bimonthly.resample('W').mean()

predicted_result_household_monthly = predicted_result_household_monthly.resample('W').mean()

predictions_household = pd.DataFrame({'mensal': predicted_result_household_monthly.groupby(predicted_result_household_monthly.notnull().cumsum()).transform(lambda x : x.sum()/len(x))['2011-04-03':'2016-05-01'],

                         'weekly': predicted_result_household_weekly['2011-04-03':'2016-05-01'],

                          'daily': predicted_result_household_daily.resample('W').sum()['2011-04-03':'2016-05-01']})

prediction_household_mean = pd.DataFrame.mean(predictions_household, axis = 1)

prediction_household_median = pd.DataFrame.median(predictions_household, axis = 1)

prediction_household_min = pd.DataFrame.min(predictions_household, axis = 1)

prediction_household_max = pd.DataFrame.max(predictions_household, axis = 1)



alpha_household_max = alpha(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_max['2011-04-03':'2015-03-29 '])

alpha_household_min = alpha(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_min['2011-04-03':'2015-03-29 '])

alpha_household_mean = alpha(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_mean['2011-04-03':'2015-03-29 '])

alpha_household_median = alpha(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_median['2011-04-03':'2015-03-29 '])

fig = go.Figure(

    data=[go.Scatter(y=data_cat_final_weekly.HOUSEHOLD, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Actual'), 

          go.Scatter(y=prediction_household_min[0:-1]*alpha_household_min, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Min'),

          go.Scatter(y=prediction_household_max[0:-1]*alpha_household_max, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Max'),

          go.Scatter(y=prediction_household_mean[0:-1]*alpha_household_mean, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict Mean'),

          go.Scatter(y=prediction_household_median[0:-1]*alpha_household_median, x= data_cat_final_weekly.HOUSEHOLD.index, name= 'Predict median')],

    layout=go.Layout(

        xaxis=dict(showgrid=False),

        yaxis=dict(showgrid=False),

    )

)

fig.update_layout(title_text="Household Category - MAPA SARIMA forecast")

fig.show()
rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[-28::], predicted_result_household_weekly[-56-1:-28-1], 'weekly Household - Test')

rmse_household_weekly= rmse(data_cat_final_weekly.HOUSEHOLD[start:-28], predicted_result_household_weekly[start:-28*2-1], 'weekly Household - Train')



#test Fold

rmse_household_max = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_max[-57:-29]*alpha_household_max, 'Household Test - max')

rmse_household_min = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_min[-57:-29]*alpha_household_min, 'Household Test - min')

rmse_household_mean = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_mean[-57:-29]*alpha_household_mean, 'Household Test - mean')

rmse_household_median = rmse(data_cat_final_weekly.HOUSEHOLD[-28::], prediction_household_median[-57:-29]*alpha_household_median, 'Household Test - median')

#train Fold

rmse_household_max = rmse(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_max['2011-04-03':'2015-03-29 ']*alpha_household_max, 'Household Train - max')

rmse_household_min = rmse(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_min['2011-04-03':'2015-03-29 ']*alpha_household_min, 'Household Train - min')

rmse_household_mean = rmse(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_mean['2011-04-03':'2015-03-29 ']*alpha_household_mean, 'Household Train - mean')

rmse_household_median = rmse(data_cat_final_weekly.HOUSEHOLD['2011-04-03':'2015-03-29 '], prediction_household_median['2011-04-03':'2015-03-29 ']*alpha_household_median, 'Household Train - median' )