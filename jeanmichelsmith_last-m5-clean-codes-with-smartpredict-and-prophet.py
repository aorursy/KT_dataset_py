!pip install FbProphet
!pip install git+https://pip_user:pvtY-b-sxREusdxJaR8f@gitlab.com/Smartpredict/smartpredict-library.git#egg=smartpredict
!pip install htsprophet
!pip install Dask
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# import smartpredict as sp #smartpredict api
import sklearn as skl
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fbprophet import Prophet
import matplotlib.pylab as plt
import htsprophet
import itertools
from datetime import datetime
from fbprophet.plot import plot_yearly


import warnings
import seaborn as sns
from itertools import cycle

# #prophet the htsp
# from htsprophet.hts import hts, orderHier, makeWeekly
# from htsprophet.htsPlot import plotNode, plotChild, plotNodeComponents

# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
stv = pd.read_csv(r'/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv',parse_dates=[0])
stv
d_cols = [c for c in stv.columns if 'd_' in c] #col d_c...
sales = stv.loc[stv['id'] == 'HOBBIES_1_001_CA_1_validation'][d_cols].T


cal = pd.read_csv(r'/kaggle/input/m5-forecasting-accuracy/calendar.csv')
cal


sales = sales.reset_index()
sales = sales.loc[:,0]
dates = cal.loc[:,'date']

frame = { 'Date': dates, 'Sales': sales }
df = pd.DataFrame(frame)
df.head()

print('-'*60)
print('*** Head of the dataframe ***')
print('-'*60)
print(df.head())
print('-'*60)
print('*** Tail of the dataframe ***')
print('-'*60)
print(df.tail())
df.info()
# Features days of the week
def date_features(df, label=None):
    df = df.copy()

    df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['month'] = df['date'].dt.strftime('%B')
    df['year'] = df['date'].dt.strftime('%Y')
    df['dayofweek'] = df['date'].dt.strftime('%A')
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
X, y = date_features(df, label='Sales')
df_new = pd.concat([X, y], axis=1)
df_new.head()
#see trends 
fig, ax = plt.subplots(figsize=(14,5))
palette = sns.color_palette("mako_r", 4)
a = sns.barplot(x="month", y="Sales",hue = 'year',data=df_new)
a.set_title("Store Sales Data",fontsize=15)
plt.legend(loc='upper right')
plt.show()
df=df.rename(columns={'Date':'ds','Sales':'y'})
df.head()
# # Detect and remove outliers
# df.loc[(df['ds'] > '2011-01-29') & (df['ds'] < '2011-06-19'), 'y'] = None
# model = Prophet().fit(df)
# fig = model.plot(model.predict(future))
# dataframes of holidays over training and forecasting periods 
SupB = pd.DataFrame ({'holiday': 'SuperBowls','ds': pd.to_datetime(['2011 - 02 - 06', '2012 - 02 - 05','2013 - 02 - 03','2014 - 02 - 02', '2015 - 02 - 01', '2016 - 02 - 07'], errors='coerce')})
ny = pd.DataFrame({'holiday': "NewYearsDay", 'ds' : pd.to_datetime(['2011 - 01 - 01','2012 - 01 - 01','2013 - 01 - 01','2014 - 01 - 01','2015 - 01 - 01'])})  
mlk = pd.DataFrame({'holiday': 'Birthday of Martin Luther King, Jr.', 'ds' : pd.to_datetime(['2012 - 01 - 16','2013 - 01 - 21','2014 - 01 - 20','2015 - 01 - 19','2016 - 01 - 18'])}) 
# wash = pd.DataFrame({'holiday': "Washington's Birthday", 'ds' : pd.to_datetime(['2016-02-15', '2017-02-20'])})
mem = pd.DataFrame({'holiday': 'Memorial Day', 'ds' : pd.to_datetime(['2011 - 05 - 30', '2012 - 05 - 28','2012 - 05 - 27','2013 - 05 - 26','2015 - 05 - 25','2016 - 05 - 30'])})
ind = pd.DataFrame({'holiday': 'Independence Day', 'ds' : pd.to_datetime(['2011 - 04 - 07','2012 - 07 - 04','2013 - 07 - 04','2014 - 07 - 04','2015 - 07 - 04'])})
lab = pd.DataFrame({'holiday': 'Labor Day', 'ds': pd.to_datetime(['2011 - 09 - 05', '2012 - 09 - 03','2013 - 09 - 02','2014 - 09 - 07'])})
col = pd.DataFrame({'holiday': 'Columbus Day', 'ds': pd.to_datetime(['2011 - 10 - 10', '2012 - 10 - 08','2013 - 10 - 14','2014 - 10 - 13','2015 - 10 - 12'])})
vet = pd.DataFrame({'holiday': "Veteran's Day", 'ds': pd.to_datetime(['2011 - 11 - 11','2012 - 11 - 11','2013 - 11 - 11','2014 - 11 - 11','2015 - 11 - 11',])})
thanks = pd.DataFrame({'holiday': 'Thanksgiving Day', 'ds' : pd.to_datetime(['2011 - 11 - 24','2012 - 11 - 22','2013 - 11 - 28','2014 - 11 - 27','2015 - 11 - 26'])})
christ = pd.DataFrame({'holiday': 'Christmas', 'ds': pd.to_datetime(['2011 - 12 - 25', '2012 - 12 - 25','2013 - 12 - 25','2014 - 12 - 25','2015 - 12 - 25'])})
MotD = pd.DataFrame({'holiday':'MothersDay', 'ds': pd.to_datetime(['2011 - 05 - 08', '2012 - 05 - 13','2013 - 05 - 12', '2014 - 05 - 11','2015 - 05 - 10','2016 - 05 - 8'])})
FatD = pd.DataFrame({'holiday': 'FathersDay', 'ds': pd.to_datetime(['2011 - 06 - 19', '2012 - 06 - 17','2013 - 06 - 16', '2015 - 06 - 21'])})
Val = pd.DataFrame({'holiday':'ValentinesDay', 'ds': pd.to_datetime (['2011 - 02 - 14', '2012 - 02 - 14', '2013 - 02 - 14','2014 - 02 - 14','2015 - 02 - 14','2016 - 02 - 14'])})
Pdt = pd.DataFrame({'holiday': 'PresidentsDay', 'ds': pd.to_datetime(['2011 - 02 - 21', '2012 - 02 - 20', '2013 - 02 - 18','2014 - 02 - 17','2015 - 02 - 16','2016 - 02 - 15'])})
East= pd.DataFrame({'holiday':'Easter', 'ds': pd.to_datetime(['2012 - 04 - 08', '2013 - 03 - 03','2014 - 02 - 04', '2015 - 04 - 05','2016 - 03 - 27'])})
Hal = pd.DataFrame({'holiday': 'Halloween', 'ds': pd.to_datetime(['2011 - 10 - 31', '2012 - 10 - 31','2013 - 10 - 31', '2014 - 10 - 31','2015 - 10 - 31'])})
LentS = pd.DataFrame({'holiday': 'LentStart', 'ds': pd.to_datetime(['2011 - 03 - 09', '2012 - 02 - 22','2013 - 02 - 13','2014 - 03 - 05','2015 - 02 - 18','2016 - 2 - 10'])})
LentW = pd.DataFrame({'holiday':'LentWeek2', 'ds': pd.to_datetime(['2011 - 03 - 16', '2012 - 02 - 29','2013 - 02 - 20', '2014 - 03 - 12','2015 - 02 - 25','2016 - 02 - 17'])})
StP = pd.DataFrame({'holiday':'StPatricksDay', 'ds': pd.to_datetime(['2011 - 03 - 17','2012 - 03 - 17','2013 - 03 - 17','2014 - 03 - 17','2015 - 03 - 17','2016 - 03 - 17'])})
Cinco = pd.DataFrame({'holiday': 'Cinco de Mayo', 'ds': pd.to_datetime(['2011 - 05 - 05', '2012 - 05 - 05','2013 - 05 - 05','2014 - 05 - 05','2015 - 05 - 05'])})
Pur = pd.DataFrame({'holiday': 'PurimEnd', 'ds': pd.to_datetime(['2011 - 03 - 20', '2012 - 08 - 03', '2013 - 02 - 24','2014 - 03 - 16','2015 - 03 - 05','2016 - 03 - 24'])})
EAAd = pd.DataFrame({'holiday': 'EidAlAdah', 'ds': pd.to_datetime(['2011 - 07 - 11','2012 - 10 - 26','2013 - 10 - 15','2014 - 10 - 4','2015 - 09 - 24'])})
Pes = pd.DataFrame({'holiday': 'Pesach', 'ds': pd.to_datetime(['2011 - 04 - 26','2012 - 04 - 14','2013 - 04 - 02','2014 - 04 - 22','2015 - 04 - 11','2016 - 04 - 30'])})
Chan = pd.DataFrame({'holiday': 'Chanukah', 'ds': pd.to_datetime(['2011 - 12 - 28','2012 - 12 - 16','2013 - 12 - 05','2015 - 12 - 14'])})
EidF = pd.DataFrame({'holiday': 'EidaLFitr', 'ds': pd.to_datetime(['2011 - 08 - 31','2012 - 08 - 19','2013 - 08- 08','2014 - 07 - 29','2015 - 07 - 18'])})
OXmas = pd.DataFrame({'holiday': 'OrthodoxChristmas', 'ds': pd.to_datetime(['2012 - 01 - 07','2013 - 01 - 07','2014 - 01 - 07','2015 - 01 - 07','2016 - 01 - 07'])})
OEast = pd.DataFrame({'holiday': 'OrthodoxEaster', 'ds': pd.to_datetime(['2011 - 04 - 24','2013 - 04 - 15','2014 - 05 - 05','2015 - 04 - 12','2016 - 01 - 05'])})
Ram = pd.DataFrame({'holiday': 'Ramadan', 'ds': pd.to_datetime(['2011 - 08 - 01','2012 - 07 - 20','2013 - 09 - 07','2014 - 06 - 29','2015 - 06 - 18','2016 - 06 - 07'])})
NBAfs = pd.DataFrame({'holiday': 'NBAFinalStarts', 'ds': pd.to_datetime(['2011 - 05 - 31','2012 - 06 - 12','2013 - 06 - 06','2014 - 06 - 05','2016 - 06 - 02'])})
NBAfe = pd.DataFrame({'holiday': 'NBAFinalEnds', 'ds': pd.to_datetime(['2011 - 06 - 12','2012 - 06 - 21', '2013 - 06 - 20','2014 - 06 - 15','2015 - 06 - 16','2016 - 06 - 19'])})

holidays = pd.concat([SupB,ny, mlk, mem, ind, lab, col, vet, thanks, christ, MotD, FatD, Val, Pdt,East, Hal, LentS, LentW, StP, Pes, OXmas, Cinco, Pur, EAAd, EidF, OEast, Ram, Chan, NBAfs, NBAfe])

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# REMOVE NEGATIVE FORECASTED VALUES/outliers
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
future.tail()
# Function to remove any negative forecasted values.
def remove_negs(ts):
    ts['yhat'] = ts['yhat'].clip_lower(0)
    ts['yhat_lower'] = ts['yhat_lower'].clip_lower(0)
    ts['yhat_upper'] = ts['yhat_upper'].clip_lower(0)
# New Prophet model taking into account the holidays
m= Prophet(holidays= holidays,uncertainty_samples=False,yearly_seasonality=True,weekly_seasonality=True, holidays_prior_scale= 0.05).fit(df)
forecast = m.predict(future)

forecast[(forecast["NewYearsDay"] + forecast ["Memorial Day"]+ forecast ['Independence Day']+ forecast ['SuperBowls']+ forecast ['Labor Day']+ forecast ['Columbus Day']+ forecast ["Veteran's Day"]+ forecast ['Thanksgiving Day']+ forecast ['Christmas']+ forecast ['MothersDay']+ forecast ['FathersDay']+ forecast ['ValentinesDay']+ forecast ['PresidentsDay']+ forecast ['Easter']+ forecast ['Halloween']+ forecast ['LentStart']+ forecast ['LentWeek2']+forecast['StPatricksDay']+forecast['Pesach']+forecast['OrthodoxChristmas']+forecast['Cinco de Mayo']+forecast['PurimEnd']+forecast['NBAFinalEnds']+forecast['NBAFinalStarts']+forecast['EidAlAdah']+forecast['EidaLFitr']+forecast['OrthodoxEaster']+forecast['Ramadan']+forecast['Chanukah']).abs() > 0][
        ['SuperBowls',"NewYearsDay","Memorial Day",'Independence Day','Labor Day','Columbus Day',"Veteran's Day",'Thanksgiving Day','Christmas','MothersDay','FathersDay','ValentinesDay','PresidentsDay','Easter','Halloween','LentStart','LentWeek2','StPatricksDay','Cinco de Mayo','PurimEnd','EidAlAdah','Pesach','Chanukah','EidaLFitr','OrthodoxChristmas','OrthodoxEaster','Ramadan','NBAFinalStarts','NBAFinalEnds']][-10:]

fig1 = m.plot_components(forecast)
#here is how to see all the included holidays
m.train_holiday_names
# The output of cross_validation is a dataframe with the true values y and the out-of-sample forecast values yhat, at each simulated forecast date and for each cutoff date
from fbprophet.diagnostics import cross_validation
cutoffs = pd.to_datetime(['2011-05-21', '2016-01-15'])
df_cv = cross_validation(m, initial = "1,941 days",cutoffs=cutoffs, period = " 30.5 days", horizon = "28 days")
df_cv.head()  
#METRICS ACCORDING TO THE HORIZON
cutoffs = pd.to_datetime(['2011-05-21'])
df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='28 days')
from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()
#  METRICS
from fbprophet.plot import plot_cross_validation_metric
fig3 = plot_cross_validation_metric(df_cv,rolling_window=0.1, metric='rmse')



# Hyperparameters tuning
import pandas as pd
import numpy as np
import itertools
param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],# None by default,Auto places
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters, growth can be linear
for params in all_params:
    m = Prophet(**params).fit(df)  # Fit model with given params
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='28 days', parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)



best_params = all_params[np.argmin(rmses)]
print(best_params)
{'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0}
#  Modify score the outliers
# def outliers_modified_z_score(ys):
#     threshold = 3.5

#     median_y = np.median(ys)
#     median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
#     modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
#                          for y in ys]
#     return np.where(np.abs(modified_z_scores) > threshold)

# # CUSTOM MODULE FOR THE PERFORMANCE 

# def getPerfomanceMetrics(m):
#   return performance_metrics(getCrossValidationData(m))

# def getCrossValidationData(m):
#  return cross_validation(m, initial='730 days', period = '31 days', horizon = '365 days')
# CUSTOM MODULE FOR THE MODELS 
# Creation of two prophet models one for seasonality 
# def run_prophet(id1,data):
#     holidays = get_holidays(id1)
#     model = Prophet(uncertainty_samples=False,
#                     holidays=holidays,
#                     weekly_seasonality = True,
#                     yearly_seasonality= True,
#                     changepoint_prior_scale = 0.5
#                    )
#     model.add_seasonality(name='yearly', period=30.5,   fourier_order=2)
   
#  Prophet model with additional regressor
# def run_prophet(id1,data):
#     holidays = get_holidays(id1)
#     model = Prophet(uncertainty_samples=False,
#                     holidays=holidays,
#                     weekly_seasonality = True,
#                     yearly_seasonality= True,
#                     changepoint_prior_scale = 0.5
#                    )
#     model.add_seasonality(name='monthly', period=30.5, fourier_order=2)
#     model.add_regressor('sell_price')
#     m.fit(df)
    
#     future = m.make_future_dataframe(periods=10)
#     future['sales'] = future['ds'].apply(run_prophet).forecast = m.predict(future)
#     forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15)
#     fig0 = m.plot (forecast)



m = Prophet(yearly_seasonality=20).fit(df)
a = plot_yearly(m)
# Disable the built-in seasonality and replace it with two seasonalities then add the column to the future dataframe for which we are making the prediction
def is_NBA_season(ds):
    date = pd.to_datetime(ds)
    return(date.month> 6 or date.month<5)
df['on_season'] = df['ds'].apply(is_NBA_season)
df['off_season'] = ~df['ds'].apply(is_NBA_season)


m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

future['on_season'] = future['ds'].apply(is_NBA_season)
future['off_season'] = ~future['ds'].apply(is_NBA_season)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
# Plot the forecast now 
f, ax = plt.subplots(1)
f.set_figheight(10)
f.set_figwidth(20)
fig2 = m.plot(forecast,ax=ax)
plt.show()

# #function to calculate in sample SMAPE scores
# def smape_fast(y_true, y_pred): #adapted from link to discussion 
#     out = 0
#     for i in range(y_true.shape[0]):
#         if (y_true[i] != None and np.isnan(y_true[i]) ==  False):
#             a = y_true[i]
#             b = y_pred[i]
#             c = a+b
#             if c == 0:
#                 continue
#             out += math.fabs(a - b) / c
#     out *= (200.0 / y_true.shape[0])
#     return out
# model = Prophet(uncertainty_samples=False,yearly_seasonality=True,weekly_seasonality=True, holidays=holidays, changepoint_prior_scale = 0.5, holidays_prior_scale=0.05).fit(df)
# future = model.make_future_dataframe(periods=31+28, freq='D', include_history=True)
# fig3 = model.plot_components(forecast)
# plt.show()




# # Additional regressors 
# def NBA_season(ds)
#     date = pd.to_datetime(ds)
#     if date.weekday()== 9 and (date.month>5 or date.month<6)
#         return 1
#     else:
#         return 0
# df['NBA_s    
#CREATE THE PROPHET MODEL TO HANDLE THE FITTING
# create the Prophet model that will receive the holidays 
# As a reminder the params below are ommitted for the ones in params grid
# model = Prophet(uncertainty_samples=False,yearly_seasonality=True,weekly_seasonality=True, holidays=holidays, changepoint_prior_scale = 0.5, holidays_prior_scale=0.05).fit(df)
# future = model.make_future_dataframe(periods=31+28, freq='D', include_history=True)
# future.tail()
# # forecast = m.fit(df).predict(future)

# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()