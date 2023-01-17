import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  matplotlib.pyplot  as  plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
### Read the time series data

univariate_series   =  pd.read_csv('/kaggle/input/time-series-data/daily-min-temperatures.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)

### Print first five records
univariate_series.head()
univariate_series.plot()
plt.ylabel('Minimum Temp')
plt.title('Minimum temperature in Southern Hemisphere \n  from 1981 to 1990')
plt.show()
# process the date time information 

from datetime import datetime
def parse(x):
    return datetime.strptime(x,'%Y %m %d %H')

# Load dataset
pollution_df = pd.read_csv("/kaggle/input/time-series-data/pollution.csv",parse_dates = [['year', 'month', 'day', 'hour']],index_col=0, date_parser=parse)
pollution_df.drop('No', axis=1, inplace=True)
# manually specify column names

pollution_df.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
pollution_df.index.name = 'date'

# mark all NA values with 0
pollution_df['pollution'].fillna(0, inplace=True)

# drop the first 24 hours
pollution_df = pollution_df[24:]

# summarize first 5 rows
print(pollution_df.head(5))

values = pollution_df.values

# specify columns to plot

groups = [0, 1, 2, 3, 5, 6, 7]
i = 1

# plot each column
plt.figure()

for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(pollution_df.columns[group], y=0.5, loc='right')
    i += 1
    
plt.show()
airPax_df = pd.read_csv('/kaggle/input/time-series-data/AirPassengers.csv')
#Parse strings to datetime type
airPax_df['Month'] = pd.to_datetime(airPax_df['Month'],infer_datetime_format=True) #convert from string to datetime
airPax_df_indexed = airPax_df.set_index(['Month'])
airPax_df_indexed.head(5)
plt.plot(airPax_df_indexed) 
plt.show()
### Save the TS object
airPax_df_indexed.to_csv('ts1.csv', index = True, sep = ',')

### Check the object retrieved
series1 = pd.read_csv('ts1.csv', header = 0)


### Check
print(type(series1))
print(series1.head(2).T)
india_gdp_df = pd.read_csv("/kaggle/input/time-series-data/GDPIndia.csv")
date_rng = pd.date_range(start='1/1/1960', end='31/12/2017', freq='A')
india_gdp_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Year'])
india_gdp_df.head(5).T
plt.plot(india_gdp_df.TimeIndex, india_gdp_df.GDPpercapita)
plt.legend(loc='best')
plt.show()
### Load as a pickle object

import pickle

with open('GDPIndia.obj', 'wb') as fp:
        pickle.dump(india_gdp_df, fp)
### Retrieve the pickle object

with open('GDPIndia.obj', 'rb') as fp:
     india_gdp1_df = pickle.load(fp)
        
india_gdp1_df.head(5).T
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
 
series = pd.read_csv('/kaggle/input/time-series-data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

series.plot()
plt.show()
### Read the time series data

series   =  pd.read_csv('/kaggle/input/time-series-data/daily-min-temperatures.csv', header = 0, index_col = 0, parse_dates = True, squeeze = True)

series.plot()
plt.ylabel('Minimum Temp')
plt.title('Minimum temperature in Southern Hemisphere \n From 1981 to 1990')
plt.show()
months         = pd.DataFrame()
one_year       = series['1990'] 
groups         = one_year.groupby(pd.Grouper(freq='M')) 
months         = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1) 
months         = pd.DataFrame(months) 
months.columns = range(1,13) 
months.boxplot() 
plt.show()
groups = series.groupby(pd.Grouper(freq='A')) 
years  = pd.DataFrame() 
for name, group in groups: 
    years[name.year] = group.values 
years.boxplot() 
plt.show()
tractor_df = pd.read_csv("/kaggle/input/time-series-data/TractorSales.csv")
tractor_df.head(5)
dates = pd.date_range(start='2003-01-01', freq='MS', periods=len(tractor_df))
import calendar
tractor_df['Month'] = dates.month
tractor_df['Month'] = tractor_df['Month'].apply(lambda x: calendar.month_abbr[x])
tractor_df['Year'] = dates.year
#Tractor.drop(['Month-Year'], axis=1, inplace=True)
tractor_df.rename(columns={'Number of Tractor Sold':'Tractor-Sales'}, inplace=True)
tractor_df = tractor_df[['Month', 'Year', 'Tractor-Sales']]
tractor_df.set_index(dates, inplace=True)
tractor_df = tractor_df[['Tractor-Sales']]
tractor_df.head(5)
tractor_df.plot()
plt.ylabel('Tractor Sales')
plt.title("Tractor Sales from 2003 to 2014")
plt.show()
months         = pd.DataFrame()
one_year       = tractor_df['2011'] 
groups         = one_year.groupby(pd.Grouper(freq='M')) 
months         = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1) 
months         = pd.DataFrame(months) 
months.columns = range(1,13) 
months.boxplot() 
plt.show()
tractor_df['2003']
turnover_df= pd.read_csv('/kaggle/input/time-series-data/RetailTurnover.csv')
date_rng = pd.date_range(start='1/7/1982', end='31/3/1992', freq='Q')
turnover_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Quarter'])
turnover_df.head()
plt.plot(turnover_df.TimeIndex, turnover_df.Turnover)
plt.legend(loc='best')
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
import  statsmodels.api as sm
decompTurnover_df = sm.tsa.seasonal_decompose(turnover_df.Turnover, model="additive", freq=4)
decompTurnover_df.plot()
plt.show()
trend = decompTurnover_df.trend
seasonal = decompTurnover_df.seasonal
residual = decompTurnover_df.resid
print(trend.head(12))
print(seasonal.head(12))
print(residual.head(12))

airPax_df = pd.read_csv('/kaggle/input/time-series-data/AirPax.csv')
print(airPax_df.head())
date_rng = pd.date_range(start='1/1/1949', end='31/12/1960', freq='M')
print(date_rng)
airPax_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Month'])
print(airPax_df.head())
decompAirPax = sm.tsa.seasonal_decompose(airPax_df.Passenger, model="multiplicative", freq=12)
decompAirPax.plot()
plt.show()
seasonal = decompAirPax.seasonal
seasonal.head(4)
quarterly_turnover = pd.pivot_table(turnover_df, values = "Turnover", columns = "Quarter", index = "Year")
quarterly_turnover
quarterly_turnover.plot()
plt.show()
quarterly_turnover.boxplot()
plt.show()
petrol_df = pd.read_csv('/kaggle/input/time-series-data/Petrol.csv')
petrol_df.head()
date_rng = pd.date_range(start='1/1/2001', end='30/9/2013', freq='Q')

#date_rng
petrol_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Quarter'])
print(petrol_df.head())

plt.plot(petrol_df.TimeIndex, petrol_df.Consumption)
plt.legend(loc='best')
plt.show()
airTemp_df =  pd.read_csv('/kaggle/input/time-series-data/AirTemp.csv')
date_rng =  pd.date_range(start='1/1/1920', end='31/12/1939', freq='M')
airTemp_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Month'])
airTemp_df.head()
plt.plot(airTemp_df.TimeIndex, airTemp_df.AvgTemp)
plt.legend(loc='best')
plt.show()
temp_avg = airTemp_df.copy()
temp_avg['avg_forecast'] = airTemp_df['AvgTemp'].mean()

plt.figure(figsize=(12,8))
plt.plot(airTemp_df['AvgTemp'], label='Data')
plt.plot(temp_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()
mvg_avg = airTemp_df.copy()
mvg_avg['moving_avg_forecast'] = airTemp_df['AvgTemp'].rolling(12).mean()
plt.plot(airTemp_df['AvgTemp'], label='Average Temperature')
plt.plot(mvg_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()
USGDP_df    = pd.read_csv('/kaggle/input/time-series-data/GDPIndia.csv', header=0)
print(USGDP_df.head())
date_rng = pd.date_range(start='1/1/1929', end='31/12/1991', freq='A')
print(date_rng)

USGDP_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Year'])
plt.plot(USGDP_df.TimeIndex, USGDP_df.GDPpercapita)

plt.legend(loc='best')
plt.show()
mvg_avg_USGDP = USGDP_df.copy()
mvg_avg_USGDP['moving_avg_forecast'] = USGDP_df['GDPpercapita'].rolling(5).mean()
plt.plot(USGDP_df['GDPpercapita'], label='US GDP')
plt.plot(mvg_avg_USGDP['moving_avg_forecast'], label='US GDP MA(5)')
plt.legend(loc='best')
plt.show()
IndiaGDP_df = pd.read_csv('/kaggle/input/time-series-data/GDPIndia.csv', header=0)

date_rng = pd.date_range(start='1/1/1960', end='31/12/2017', freq='A')
IndiaGDP_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Year'])

print(IndiaGDP_df.head())

plt.plot(IndiaGDP_df.TimeIndex, IndiaGDP_df.GDPpercapita)
plt.legend(loc='best')
plt.show()
mvg_avg_IndiaGDP = IndiaGDP_df.copy()
mvg_avg_IndiaGDP['moving_avg_forecast'] = IndiaGDP_df['GDPpercapita'].rolling(3).mean()

plt.plot(IndiaGDP_df['GDPpercapita'], label='India GDP per Capita')
plt.plot(mvg_avg_IndiaGDP['moving_avg_forecast'], label='India GDP/Capita MA(3)')
plt.legend(loc='best')
plt.show()
import pandas as pd
import numpy  as np
s = pd.Series([1,2,3,4,5,6])
s.loc[4] = np.NaN
print(s)
def handle_missing_values():
    print()
    print(format('How to deal with missing values in a Timeseries in Python',
                 '*^82'))
    
    # Create date
    time_index = pd.date_range('28/03/2017', periods=5, freq='M')

    # Create data frame, set index
    df = pd.DataFrame(index=time_index);
    print(df)

    # Create feature with a gap of missing values
    df['Sales'] = [1.0,2.0,np.nan,np.nan,5.0];
    print(); print(df)

    # Interpolate missing values
    df1= df.interpolate();
    print(); print(df1)

    # Forward-fill Missing Values
    df2 = df.ffill();
    print(); print(df2)

    # Backfill Missing Values
    df3 = df.bfill();
    print(); print(df3)

    # Interpolate Missing Values But Only Up One Value
    df4 = df.interpolate(limit=1, limit_direction='forward');
    print(); print(df4)

    # Interpolate Missing Values But Only Up Two Values
    df5 = df.interpolate(limit=2, limit_direction='forward');
    print(); print(df5)
handle_missing_values()
waterConsumption_df=pd.read_csv("/kaggle/input/time-series-data/WaterConsumption.csv")
waterConsumption_df.head()
# Converting the column to DateTime format
waterConsumption_df.Date = pd.to_datetime(waterConsumption_df.Date, format='%d-%m-%Y')
waterConsumption_df = waterConsumption_df.set_index('Date')
waterConsumption_df.head()
# For charting purposes, we will add a column that contains the missing values only.

waterConsumption_df = waterConsumption_df.assign(missing= np.nan)
waterConsumption_df.missing[waterConsumption_df.target.isna()] = waterConsumption_df.reference
waterConsumption_df.info()
waterConsumption_df.plot(style=['k--', 'bo-', 'r*'],figsize=(20, 10))
# Filling using mean or median
# Creating a column in the dataframe
# instead of : df['NewCol']=0, we use
# df = df.assign(NewCol=default_value)
# to avoid pandas warning.
waterConsumption_df = waterConsumption_df.assign(FillMean=waterConsumption_df.target.fillna(waterConsumption_df.target.mean()))
waterConsumption_df = waterConsumption_df.assign(FillMedian=waterConsumption_df.target.fillna(waterConsumption_df.target.median()))
# imputing using the rolling average
waterConsumption_df = waterConsumption_df.assign(RollingMean=waterConsumption_df.target.fillna(waterConsumption_df.target.rolling(24,min_periods=1,).mean()))
# imputing using the rolling median
waterConsumption_df = waterConsumption_df.assign(RollingMedian=waterConsumption_df.target.fillna(waterConsumption_df.target.rolling(24,min_periods=1,).median()))# imputing using the median
#Imputing using interpolation with different methods

waterConsumption_df = waterConsumption_df.assign(InterpolateLinear=waterConsumption_df.target.interpolate(method='linear'))
waterConsumption_df = waterConsumption_df.assign(InterpolateTime=waterConsumption_df.target.interpolate(method='time'))
waterConsumption_df = waterConsumption_df.assign(InterpolateQuadratic=waterConsumption_df.target.interpolate(method='quadratic'))
waterConsumption_df = waterConsumption_df.assign(InterpolateCubic=waterConsumption_df.target.interpolate(method='cubic'))
waterConsumption_df = waterConsumption_df.assign(InterpolateSLinear=waterConsumption_df.target.interpolate(method='slinear'))
waterConsumption_df = waterConsumption_df.assign(InterpolateAkima=waterConsumption_df.target.interpolate(method='akima'))
waterConsumption_df = waterConsumption_df.assign(InterpolatePoly5=waterConsumption_df.target.interpolate(method='polynomial', order=5)) 
waterConsumption_df = waterConsumption_df.assign(InterpolatePoly7=waterConsumption_df.target.interpolate(method='polynomial', order=7))
waterConsumption_df = waterConsumption_df.assign(InterpolateSpline3=waterConsumption_df.target.interpolate(method='spline', order=3))
waterConsumption_df = waterConsumption_df.assign(InterpolateSpline4=waterConsumption_df.target.interpolate(method='spline', order=4))
waterConsumption_df = waterConsumption_df.assign(InterpolateSpline5=waterConsumption_df.target.interpolate(method='spline', order=5))
#Scoring the results and see which is better

# Import a scoring metric to compare methods
from sklearn.metrics import r2_score

results = [(method, r2_score(waterConsumption_df.reference, waterConsumption_df[method])) for method in list(waterConsumption_df)[3:]]
results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])
results_df.sort_values(by='R_squared', ascending=False)
#data after imputation

final_df= waterConsumption_df[['reference', 'target', 'missing', 'InterpolateTime' ]]
final_df.plot(style=['b-.', 'ko', 'r.', 'rx-'], figsize=(20,10));
plt.ylabel('Temperature');
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=True, ncol=5, prop={'size': 14} );

def parser(x):
       return datetime.strptime('190'+x, '%Y-%m')

shampoo_df = pd.read_csv('/kaggle/input/time-series-data/shampoo.csv', header = 0, index_col = 0, parse_dates = True, 
                               squeeze = True, date_parser = parser)

upsampled_ts = shampoo_df.resample('D').mean()
print(upsampled_ts .head(36))
interpolated = upsampled_ts.interpolate(method = 'linear')
interpolated.plot()
plt.show()
interpolated1 = upsampled_ts.interpolate(method = 'spline', order = 2)
interpolated1.plot()
plt.show()
print(interpolated1.head(12))
resample = shampoo_df.resample('Q')
quarterly_mean_sales = resample.mean()
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
plt.show()
resample = shampoo_df.resample('A')
yearly_mean_sales = resample.sum()
print(yearly_mean_sales.head() )
yearly_mean_sales.plot()
plt.show()
def parser(x): 
    return datetime.strptime('190'+x, '%Y-%m')

shampoo_df=pd.read_csv("/kaggle/input/time-series-data/shampoo.csv",header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)

X = shampoo_df.values 
diff = list() 
for i in range(1, len(X)): 
     value = X[i] - X[i - 1] 
     diff.append(value) 
plt.plot(diff) 
plt.show()
from sklearn.linear_model import LinearRegression 

# fit linear model 
X = [i for i in range(0, len(shampoo_df))] 
X = np.reshape(X, (len(X), 1))
y = shampoo_df.values 
model = LinearRegression() 
model.fit(X, y) 

# calculate trend 
trend = model.predict(X) 

# plot trend 
plt.plot(y) 
plt.plot(trend) 
plt.show() 

# detrend 
detrended = [y[i]-trend[i] for i in range(0, len(shampoo_df))] 

# plot detrended 
plt.plot(detrended) 
plt.show()
# deseasonalize monthly data by differencing 

min_temperature = pd.read_csv('/kaggle/input/time-series-data/daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
resample       = min_temperature.resample('M') 
monthly_mean   = resample.mean() 

X = min_temperature.values 
diff = list() 
months_in_year = 12 

for i in range(months_in_year, len(monthly_mean)): 
    value = monthly_mean[i] - monthly_mean[i - months_in_year] 
    diff.append(value) 

plt.plot(diff) 
plt.show()
from sklearn.metrics import  mean_squared_error
def MAE(y,yhat):
    diff = np.abs(np.array(y)-np.array(yhat))
    try:
        mae =  round(np.mean(np.fabs(diff)),3)
    except:
        print("Error while calculating")
        mae = np.nan
    return mae
def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.mean(np.abs((y - yhat) / y)) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape
female_birth_series =  pd.read_csv('/kaggle/input/time-series-data/daily-total-female-births.csv', header=0, index_col=0, parse_dates=True, squeeze=True) 

# tail rolling average transform
rolling =  female_birth_series.rolling(window = 3) # arbitrarily chosen

rolling_mean = rolling.mean()
female_birth_series.plot()

rolling_mean.plot(color = 'red')
plt.show()

# Zoomed plot original and transformed dataset
female_birth_series[:100].plot()
rolling_mean[:100].plot(color = 'red')
plt.show()
y_df = pd.DataFrame( {'Observed':female_birth_series.values, 'Predicted':rolling_mean})
y_df .dropna(axis = 0, inplace = True)
print(y_df.tail())

rmse = np.sqrt(mean_squared_error(y_df.Observed, y_df.Predicted))
print("\n\n Accuracy measures ")
print('RMSE: %.3f' % rmse)
n = y_df.shape[0]

mae = MAE(y_df.Observed, y_df.Predicted)
print('MAE: %d' % np.float(mae))

mape = MAPE(y_df.Observed, y_df.Predicted)
print('MAPE: %.3f' % np.float(mape))
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa
from   sklearn.metrics                 import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing
def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape
petrol_df =  pd.read_csv('/kaggle/input/time-series-data/Petrol.csv')
date_rng  =  pd.date_range(start='1/1/2001', end='30/9/2013', freq='Q')
print(date_rng)
petrol_df['TimeIndex'] = pd.DataFrame(date_rng, columns=['Quarter'])
print(petrol_df.head(3).T)

plt.plot(petrol_df.TimeIndex, petrol_df.Consumption)
plt.title('Original data before split')
plt.show()
#Creating train and test set 

train = petrol_df[0:int(len(petrol_df)*0.7)] 
test= petrol_df[int(len(petrol_df)*0.7):]

print("\n Training data start at \n")
print (train[train.TimeIndex == train.TimeIndex.min()],['Year','Quarter'])
print("\n Training data ends at \n")
print (train[train.TimeIndex == train.TimeIndex.max()],['Year','Quarter'])

print("\n Test data start at \n")
print (test[test.TimeIndex == test.TimeIndex.min()],['Year','Quarter'])

print("\n Test data ends at \n")
print (test[test.TimeIndex == test.TimeIndex.max()],['Year','Quarter'])

plt.plot(train.TimeIndex, train.Consumption, label = 'Train')
plt.plot(test.TimeIndex, test.Consumption,  label = 'Test')
plt.legend(loc = 'best')
plt.title('Original data after split')
plt.show()
# create class
model = SimpleExpSmoothing(np.asarray(train['Consumption']))
# fit model

alpha_list = [0.1, 0.5, 0.99]

pred_SES = test.copy() # Have a copy of the test dataset

for alpha_value in alpha_list:

    alpha_str =  "SES" + str(alpha_value)
    mode_fit_i  =  model.fit(smoothing_level = alpha_value, optimized=False)
    pred_SES[alpha_str]  =  mode_fit_i.forecast(len(test['Consumption']))
    rmse =  np.sqrt(mean_squared_error(test['Consumption'], pred_SES[alpha_str]))
    mape =  MAPE(test['Consumption'],pred_SES[alpha_str])
###
    print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse, mape))
    plt.figure(figsize=(16,8))
    plt.plot(train.TimeIndex, train['Consumption'], label ='Train')
    plt.plot(test.TimeIndex, test['Consumption'], label  ='Test')
    plt.plot(test.TimeIndex, pred_SES[alpha_str], label  = alpha_str)
    plt.title('Simple Exponential Smoothing with alpha ' + str(alpha_value))
    plt.legend(loc='best') 
    plt.show()
pred_opt   =  SimpleExpSmoothing(train['Consumption']).fit(optimized = True)
print('')
print('== Simple Exponential Smoothing ')
print('')

print('')
print('Smoothing Level', np.round(pred_opt.params['smoothing_level'], 4))
print('Initial Level',   np.round(pred_opt.params['initial_level'], 4))
print('')

y_pred_opt           = pred_opt.forecast(steps = 16)
df_pred_opt          = pd.DataFrame({'Y_hat':y_pred_opt,'Y':test['Consumption'].values})

rmse_opt             =  np.sqrt(mean_squared_error(test['Consumption'], y_pred_opt))
mape_opt             =  MAPE(test['Consumption'], y_pred_opt)

alpha_value          = np.round(pred_opt.params['smoothing_level'], 4)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))

plt.figure(figsize=(16,8))
plt.plot(train.TimeIndex, train['Consumption'], label = 'Train')
plt.plot(test.TimeIndex, test['Consumption'],  label = 'Test')
plt.plot(test.TimeIndex, y_pred_opt,           label = 'SES_OPT')
plt.title('Simple Exponential Smoothing with alpha ' + str(alpha_value))
plt.legend(loc='best') 
plt.show()

print(df_pred_opt.head().T)
from   statsmodels.tsa.holtwinters import  Holt
model = Holt(np.asarray(train['Consumption']))

model_fit = model.fit()

print('')
print('==Holt model Exponential Smoothing Parameters ==')
print('')
alpha_value = np.round(model_fit.params['smoothing_level'], 4)
print('Smoothing Level', alpha_value )
print('Smoothing Slope', np.round(model_fit.params['smoothing_slope'], 4))
print('Initial Level',   np.round(model_fit.params['initial_level'], 4))
print('')
Pred_Holt = test.copy()

Pred_Holt['Opt'] = model_fit.forecast(len(test['Consumption']))
plt.figure(figsize=(16,8))
plt.plot(train['Consumption'], label='Train')
plt.plot(test['Consumption'], label='Test')
plt.plot(Pred_Holt['Opt'], label='HoltOpt')
plt.legend(loc='best')
plt.show()
df_pred_opt =  pd.DataFrame({'Y_hat':Pred_Holt['Opt'] ,'Y':test['Consumption'].values})
rmse_opt =  np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt =  MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))
print(model_fit.params)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

pred1 = ExponentialSmoothing(np.asarray(train['Consumption']), trend='additive', damped=False, seasonal='additive',
                                  seasonal_periods = 12).fit() #[:'2017-01-01']
print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred1.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', np.round(pred1.params['smoothing_slope'], 4))
print('Smoothing Seasonal: ', np.round(pred1.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred1.params['initial_level'], 4))
print('Initial Slope: ', np.round(pred1.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(pred1.params['initial_seasons'], 4))
print('')

### Forecast for next 16 months

y_pred1 =  pred1.forecast(steps = 16)
df_pred1 = pd.DataFrame({'Y_hat':y_pred1,'Y':test['Consumption']})
print(df_pred1)
### Plot

fig2, ax = plt.subplots()
ax.plot(df_pred1.Y, label='Original')
ax.plot(df_pred1.Y_hat, label='Predicted')

plt.legend(loc='upper left')
plt.title('Holt-Winters Additive ETS(A,A,A) Method 1')
plt.ylabel('Qty')
plt.xlabel('Date')
plt.show()
rmse    =  np.sqrt(mean_squared_error(df_pred1.Y, df_pred1.Y_hat))
mape    =  MAPE(df_pred1.Y, df_pred1.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse, mape))
print(pred1.params)
AirPax =  pd.read_csv('/kaggle/input/time-series-data/AirPax.csv')
date_rng = pd.date_range(start='1/1/1949', end='31/12/1960', freq='M')
print(date_rng)

AirPax['TimeIndex'] = pd.DataFrame(date_rng, columns=['Month'])
print(AirPax.head())
#Creating train and test set 

train = AirPax[0:int(len(AirPax)*0.7)] 
test= AirPax[int(len(AirPax)*0.7):]
print("\n Training data start at \n")
print (train[train.TimeIndex == train.TimeIndex.min()],['Year','Month'])
print("\n Training data ends at \n")
print (train[train.TimeIndex == train.TimeIndex.max()],['Year','Month'])

print("\n Test data start at \n")
print (test[test.TimeIndex == test.TimeIndex.min()],['Year','Month'])

print("\n Test data ends at \n")
print (test[test.TimeIndex == test.TimeIndex.max()],['Year','Month'])

plt.plot(train.TimeIndex, train.Passenger, label = 'Train')
plt.plot(test.TimeIndex, test.Passenger,  label = 'Test')
plt.legend(loc = 'best')
plt.title('Original data after split')
plt.show()
pred = ExponentialSmoothing(np.asarray(train['Passenger']),
                                  seasonal_periods=12 ,seasonal='add').fit(optimized=True)

print(pred.params)

print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', np.round(pred.params['smoothing_slope'], 4))
print('Smoothing Seasonal: ', np.round(pred.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred.params['initial_level'], 4))
print('Initial Slope: ', np.round(pred.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(pred.params['initial_seasons'], 4))
print('')
pred_HoltW = test.copy()
pred_HoltW['HoltW'] = model_fit.forecast(len(test['Passenger']))
plt.figure(figsize=(16,8))
plt.plot(train['Passenger'], label='Train')
plt.plot(test['Passenger'], label='Test')
plt.plot(pred_HoltW['HoltW'], label='HoltWinters')
plt.title('Holt-Winters Additive ETS(A,A,A) Parameters:\n  alpha = ' + 
          str(alpha_value) + '  Beta:' + 
          str(np.round(pred.params['smoothing_slope'], 4)) +
          '  Gamma: ' + str(np.round(pred.params['smoothing_seasonal'], 4)))
plt.legend(loc='best')
plt.show()
df_pred_opt =  pd.DataFrame({'Y_hat':pred_HoltW['HoltW'] ,'Y':test['Passenger'].values})

rmse_opt    =  np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt    =  MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))
pred = ExponentialSmoothing(np.asarray(train['Passenger']),
                                  seasonal_periods=12 ,seasonal='multiplicative').fit(optimized=True)

print(pred.params)

print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', np.round(pred.params['smoothing_slope'], 4))
print('Smoothing Seasonal: ', np.round(pred.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred.params['initial_level'], 4))
print('Initial Slope: ', np.round(pred.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(pred.params['initial_seasons'], 4))
print('')
pred_HoltW = test.copy()

pred_HoltW['HoltWM'] = pred.forecast(len(test['Passenger']))
plt.figure(figsize=(16,8))
plt.plot(train['Passenger'], label='Train')
plt.plot(test['Passenger'], label='Test')
plt.plot(pred_HoltW['HoltWM'], label='HoltWinters')
plt.title('Holt-Winters Multiplicative ETS(A,A,M) Parameters:\n  alpha = ' + 
          str(alpha_value) + '  Beta:' + 
          str(np.round(pred.params['smoothing_slope'], 4)) +
          '  Gamma: ' + str(np.round(pred.params['smoothing_seasonal'], 4)))
plt.legend(loc='best')
plt.show()
df_pred_opt =  pd.DataFrame({'Y_hat':pred_HoltW['HoltWM'] ,'Y':test['Passenger'].values})

rmse_opt    =  np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt    =  MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))
#Importing data
def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')
 
shampoo_df = pd.read_csv('/kaggle/input/time-series-data/shampoo.csv', header=0, parse_dates=True, squeeze=True, date_parser=parser)

print(shampoo_df.head())
shampoo_df.plot()

# Creating train and test set

train    =   shampoo_df[0:int(len(shampoo_df)*0.7)] 
test     =   shampoo_df[int(len(shampoo_df)*0.7):]
### Plot data

train['Sales'].plot(figsize=(15,8), title= 'Monthly Sales', fontsize=14)
test['Sales'].plot(figsize=(15,8), title= 'Monthly Sales', fontsize=14)
shampoo_df.head()
shampoo_df1         =   shampoo_df.copy() # Make a copy

time        = [i+1 for i in range(len(shampoo_df))]
shampoo_df1['time'] = time
monthDf     = shampoo_df1[['Month']]

shampoo_df1.drop('Month', axis=1, inplace=True)
shampoo_df1.head(2)
#Creating train and test set 
train=shampoo_df1[0:int(len(shampoo_df1)*0.7)] 
test=shampoo_df1[int(len(shampoo_df1)*0.7):]
x_train = train.drop('Sales', axis=1)
x_test  = test.drop('Sales', axis=1)
y_train = train[['Sales']]
y_test  = test[['Sales']]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

predictions         = model.predict(x_test)
y_test['RegOnTime'] = predictions

plt.figure(figsize=(16,8))
plt.plot( train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(y_test['RegOnTime'], label='Regression On Time')
plt.legend(loc='best')
from math import sqrt
rmse = sqrt(mean_squared_error(test.Sales, y_test.RegOnTime))
rmse = round(rmse, 3)
mape = MAPE(test.Sales, y_test.RegOnTime)
print("For RegressionOnTime,  RMSE is %3.3f MAPE is %3.2f" %(rmse, mape))
resultsDf = pd.DataFrame({'Method':['RegressionOnTime'], 'rmse': [rmse], 'mape' : [mape]})
resultsDf
time = [i+1 for i in range(len(shampoo_df))]
shampoo_df1 = shampoo_df.copy()
shampoo_df1['time'] = time
print(shampoo_df1.head())
print(shampoo_df1.shape[0])
monthSeasonality = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']
shampoo_df1['monthSeasonality'] = monthSeasonality * 3
shampoo_df1.head()
monthDf = shampoo_df1[['Month']]
shampoo_df1.drop('Month', axis=1, inplace=True)
shampoo_df1Complete = pd.get_dummies(shampoo_df1, drop_first=True)
shampoo_df1Complete.head(2).T
#Creating train and test set 
train=shampoo_df1Complete[0:int(len(shampoo_df1Complete)*0.7)] 
test=shampoo_df1Complete[int(len(shampoo_df1Complete)*0.7):]
x_train  = train.drop('Sales', axis=1)
x_test   = test.drop('Sales', axis=1)
y_train  = train[['Sales']]
y_test   = test[['Sales']]
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
y_test['RegOnTimeSeasonal'] = predictions
plt.figure(figsize=(16,8))
plt.plot( train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(y_test['RegOnTimeSeasonal'], label='Regression On Time With Seasonal Components')
plt.legend(loc='best')
rmse = sqrt(mean_squared_error(test.Sales, y_test.RegOnTimeSeasonal))
rmse = round(rmse, 3)
mape = MAPE(test.Sales, y_test.RegOnTimeSeasonal)
print("For RegOnTimeSeasonal,  RMSE is %3.3f MAPE is %3.2f" %(rmse, mape))
tempResultsDf = pd.DataFrame({'Method':['RegressionOnTimeSeasonal'], 'rmse': [rmse], 'mape' : [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf
dd= np.asarray(train.Sales)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(12,8))
plt.plot(train.index, train['Sales'], label='Train')
plt.plot(test.index,test['Sales'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
rmse = sqrt(mean_squared_error(test.Sales, y_hat.naive))
rmse = round(rmse, 3)
mape = MAPE(test.Sales, y_hat.naive)
print("For Naive model,  RMSE is %3.3f MAPE is %3.2f" %(rmse, mape))
tempResultsDf = pd.DataFrame({'Method':['Naive model'], 'rmse': [rmse], 'mape' : [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['Sales'].mean()
plt.figure(figsize=(12,8))
plt.plot(train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
rmse = sqrt(mean_squared_error(test.Sales, y_hat_avg.avg_forecast))
rmse = round(rmse, 3)
mape = MAPE(test.Sales, y_hat_avg.avg_forecast)
print("For Simple Average model,  RMSE is %3.3f MAPE is %3.2f" %(rmse, mape))
tempResultsDf = pd.DataFrame({'Method':['Simple Average'], 'rmse': [rmse], 'mape' : [mape]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
resultsDf
shampoo_df1 = shampoo_df.copy()
shampoo_df1['moving_avg_forecast_4']  = shampoo_df['Sales'].rolling(4).mean()
shampoo_df1['moving_avg_forecast_6']  = shampoo_df['Sales'].rolling(6).mean()
shampoo_df1['moving_avg_forecast_8']  = shampoo_df['Sales'].rolling(8).mean()
shampoo_df1['moving_avg_forecast_12'] = shampoo_df['Sales'].rolling(12).mean()
cols = ['moving_avg_forecast_4','moving_avg_forecast_6','moving_avg_forecast_8','moving_avg_forecast_12']

#Creating train and test set 
train=shampoo_df1[0:int(len(shampoo_df1)*0.7)] 
test=shampoo_df1[int(len(shampoo_df1)*0.7):]

y_hat_avg = test.copy()

for col_name in cols:
    
    plt.figure(figsize=(16,8))
    plt.plot(train['Sales'], label='Train')
    plt.plot(test['Sales'], label='Test')
    plt.plot(y_hat_avg[col_name], label = col_name)
    plt.legend(loc = 'best')

    rmse = sqrt(mean_squared_error(test.Sales, y_hat_avg[col_name]))
    rmse = round(rmse, 3)
    mape = MAPE(test.Sales, y_hat_avg[col_name])
    print("For Simple Average model, %s  RMSE is %3.3f MAPE is %3.2f" %(col_name, rmse, mape))

    tempResultsDf = pd.DataFrame({'Method':[col_name], 'rmse': [rmse], 'mape' : [mape]})
    resultsDf = pd.concat([resultsDf, tempResultsDf])
print(resultsDf)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
# create class
model = SimpleExpSmoothing(train['Sales'])
model_fit = model.fit(optimized = True)
print('')
print('== Simple Exponential Smoothing ')
print('')

print('')
print('Smoothing Level', np.round(model_fit.params['smoothing_level'], 4))
print('Initial Level',   np.round(model_fit.params['initial_level'], 4))
print('')
y_hat_avg['SES']= model_fit.forecast(len(test['Sales']))

alpha_value = np.round(model_fit.params['smoothing_level'], 4)


plt.figure(figsize=(16,8))
plt.plot(train.index, train['Sales'], label = 'Train')
plt.plot(test.index, test['Sales'],   label = 'Test')
plt.plot(test.index, y_hat_avg.SES,   label = 'SES_OPT')
plt.title('Simple Exponential Smoothing with alpha ' + str(alpha_value))
plt.legend(loc='best') 
plt.show()
rmse_opt=  np.sqrt(mean_squared_error(test['Sales'], y_hat_avg.SES))
mape_opt=  MAPE(test['Sales'], y_hat_avg.SES)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))
tempResultsDf = pd.DataFrame({'Method': 'SES', 'rmse': [rmse_opt], 'mape' : [mape_opt]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
print(resultsDf)
import statsmodels.api as sm
y_hat_avg = test.copy()
model_fit = Holt(np.asarray(train['Sales'])).fit()
y_hat_avg['Holt_linear'] = model_fit.forecast(len(test))
print('')
print('==Holt model Exponential Smoothing Parameters ==')
print('')
alpha_value = np.round(model_fit.params['smoothing_level'], 4)
beta_value  = np.round(model_fit.params['smoothing_slope'], 4)

print('Smoothing Level', alpha_value )
print('Smoothing Slope', beta_value)
print('Initial Level',   np.round(model_fit.params['initial_level'], 4))
print('')
plt.figure(figsize=(16,8))
plt.plot(train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()
rmse_opt             =  np.sqrt(mean_squared_error(test['Sales'], y_hat_avg['Holt_linear']))
mape_opt             =  MAPE(test['Sales'], y_hat_avg['Holt_linear'])

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))
tempResultsDf = pd.DataFrame({'Method': 'Holt_linear', 'rmse': [rmse_opt], 'mape' : [mape_opt]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
print(resultsDf)
y_hat_avg = test.copy()
model_fit = ExponentialSmoothing(np.asarray(train['Sales']) ,seasonal_periods = 12 ,trend='add', seasonal='add').fit()
y_hat_avg['Holt_Winter'] = model_fit.forecast(len(test))

print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred.params['smoothing_level'], 4)
beta_value  = np.round(model_fit.params['smoothing_slope'], 4)
gamma_value = np.round(model_fit.params['smoothing_seasonal'], 4) 

print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', beta_value)
print('Smoothing Seasonal: ', gamma_value)
print('Initial Level: ', np.round(model_fit.params['initial_level'], 4))
print('Initial Slope: ', np.round(model_fit.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(model_fit.params['initial_seasons'], 4))
print('')
plt.figure(figsize=(16,8))
plt.plot( train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.title('Holt-Winters Multiplicative ETS(A,A,M) Parameters:\n  alpha = ' + 
          str(alpha_value) + '  Beta:' + 
          str(beta_value) +
          '  Gamma: ' + str(gamma_value))
plt.legend(loc='best')
rmse_opt=  np.sqrt(mean_squared_error(test['Sales'], y_hat_avg['Holt_Winter']))
mape_opt=  MAPE(test['Sales'], y_hat_avg['Holt_Winter'])

print("For alpha = %1.2f, beta = %1.2f, gamma = %1.2f, RMSE is %3.4f MAPE is %3.2f" %(alpha_value, beta_value, gamma_value, rmse_opt, mape_opt))
tempResultsDf = pd.DataFrame({'Method': 'Holt_Winter', 'rmse': [rmse_opt], 'mape' : [mape_opt]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
print(resultsDf)
y_hat_avg = test.copy()
model_fit = ExponentialSmoothing(np.asarray(train['Sales']) ,seasonal_periods = 12 ,trend='add', seasonal='Multiplicative').fit()
y_hat_avg['Holt_Winter_M'] = model_fit.forecast(len(test))
print('')
print('== Holt-Winters Additive ETS(A,A,M) Parameters ==')
print('')
alpha_value = np.round(pred.params['smoothing_level'], 4)
beta_value  = np.round(model_fit.params['smoothing_slope'], 4)
gamma_value = np.round(model_fit.params['smoothing_seasonal'], 4) 

print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', beta_value)
print('Smoothing Seasonal: ', gamma_value)
print('Initial Level: ', np.round(model_fit.params['initial_level'], 4))
print('Initial Slope: ', np.round(model_fit.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(model_fit.params['initial_seasons'], 4))
print('')
plt.figure(figsize=(16,8))
plt.plot( train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(y_hat_avg['Holt_Winter_M'], label='Holt_Winter_M')
plt.title('Holt-Winters Multiplicative  Parameters:\n  alpha = ' + 
          str(alpha_value) + '  Beta:' + 
          str(beta_value) +
          '  Gamma: ' + str(gamma_value))
plt.legend(loc='best')
rmse_opt=  np.sqrt(mean_squared_error(test['Sales'], y_hat_avg['Holt_Winter_M']))
mape_opt=  MAPE(test['Sales'], y_hat_avg['Holt_Winter_M'])

print("For alpha = %1.2f, beta = %1.2f, gamma = %1.2f, RMSE is %3.4f MAPE is %3.2f" %(alpha_value, beta_value, gamma_value, rmse_opt, mape_opt))
tempResultsDf = pd.DataFrame({'Method': 'Holt_Winter M', 'rmse': [rmse_opt], 'mape' : [mape_opt]})
resultsDf = pd.concat([resultsDf, tempResultsDf])
print(resultsDf)
from   statsmodels.tsa.stattools  import  adfuller
data= pd.read_csv('/kaggle/input/time-series-data/TractorSales.csv', header=0, parse_dates=[0], squeeze=True)

dates= pd.date_range(start='2003-01-01', freq='MS', periods=len(data))

data['Month']= dates.month
data['Month']= data['Month'].apply(lambda x: calendar.month_abbr[x])
data['Year']= dates.year

data.drop(['Month-Year'], axis=1, inplace=True)
data.rename(columns={'Number of Tractor Sold':'Tractor-Sales'}, inplace=True)

data= data[['Month', 'Year', 'Tractor-Sales']]
data.set_index(dates, inplace=True)

sales_ts = data['Tractor-Sales']

result = adfuller(sales_ts) 

print('ADF Statistic: %f' % result[0]) 
print('p-value: %f' % result[1]) 
sales_ts_diff   = sales_ts - sales_ts.shift(periods=1)
sales_ts_diff.dropna(inplace=True)

result = adfuller(sales_ts_diff) 

pval              = result[1]
print('ADF Statistic: %f' % result[0]) 
print('p-value: %f' % result[1]) 

if pval < 0.05:
    print('Data is stationary')
else:
    print('Data after differencing is not stationary; so try log diff')
    sales_ts_log      = np.log10(sales_ts)
    sales_ts_log.dropna(inplace=True)
    sales_ts_log_diff = sales_ts_log.diff(periods=1)
    sales_ts_log_diff.dropna(inplace=True)
    result            = adfuller(sales_ts_log_diff) 

    pval              = result[1]
    print('ADF Statistic: %f' % result[0]) 
    print('p-value: %f' % result[1]) 
    if pval < 0.05:
        print('Data after log differencing is stationary')
    else:
        print('Data after log differencing is not stationary; try second order differencing')
        sales_ts_log_diff2 = sales_ts_log.diff(periods = 2)
        sales_ts_log_diff2.dropna(inplace=True)
        result         =   adfuller(sales_ts_log_diff2) 
        pval              = result[1]
        print('ADF Statistic: %f' % result[0]) 
        print('p-value: %f' % result[1]) 
        if pval < 0.05:
            print('Data after log differencing 2nd order is stationary')
        else:
            print('Data after log differencing 2nd order is not stationary')
        
#ACF and PACF plots:
from   statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt


lag_acf    =   acf(sales_ts_log_diff2,   nlags=20)
lag_pacf   =   pacf(sales_ts_log_diff2, nlags=20, method='ols')

#Plot ACF: 

plt.figure(figsize = (15,5))
plt.subplot(121) 
plt.stem(lag_acf)
plt.axhline(y = 0, linestyle='--',color='black')
plt.axhline(y = -1.96/np.sqrt(len(sales_ts_log_diff2)),linestyle='--',color='gray')
plt.axhline(y = 1.96/np.sqrt(len(sales_ts_log_diff2)),linestyle='--',color='gray')
plt.xticks(range(0,22,1))
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.title('Autocorrelation Function')
#Plot PACF:

plt.subplot(122)
plt.stem(lag_pacf)
plt.axhline(y = 0, linestyle = '--', color = 'black')
plt.axhline(y =-1.96/np.sqrt(len(sales_ts_log_diff2)), linestyle = '--', color = 'gray')
plt.axhline(y = 1.96/np.sqrt(len(sales_ts_log_diff2)),linestyle = '--', color = 'gray')
plt.xlabel('Lag')
plt.xticks(range(0,22,1))
plt.ylabel('PACF')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()



plt.show()
from random import  seed, random
seed(1) 
random_walk = list() 
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000): 
    movement = -1 if random() < 0.5 else 1 
    value    = random_walk[i-1] + movement 
    random_walk.append(value) 
    
pd.plotting.autocorrelation_plot(random_walk) 
plt.show()

### Check stationary property

result = adfuller(random_walk) 

print('ADF Statistic: %f' % result[0]) 
print('p-value: %f' % result[1]) 
print('Critical Values:') 
for key, value in result[4].items(): 
    print('\t%s: %.3f' % (key, value))
# prepare dataset for predicting a random walk

training_size     = int(len(random_walk) * 0.70) 
training, test    = random_walk[0 : training_size], random_walk[training_size:]  

predictions       = list() 
hist              = training[-1] 

for i in range(len(test)): 
    yhat = hist 
    predictions.append(yhat) 
    hist = test[i] 
    
rmse = np.sqrt(mean_squared_error(test, predictions))  
print('\n\nPredicting a Random Walk \n RMSE: %.3f' % rmse)
# Generate
seed(1234)
rw_steps  = np.random.normal(loc = 0.001, scale = 0.01, size = 1000) + 1

### Initialize first element to 1
rw_steps[0] = 1

### Simulate the stock price

Price = rw_steps * np.cumprod(rw_steps)
Price = Price * 100

### Plot the simulated stock prices
plt.plot(rw_steps)
plt.title("Simulated random walk with drift")
plt.show()
### Check stationary property

result = adfuller(Price) 

print('ADF Statistic: %f' % result[0]) 
print('p-value: %f' % result[1]) 
print('Critical Values:') 

for key, value in result[4].items(): 
    print('\t%s: %.3f' % (key, value))
### Prediction

training_size     = int(len(Price) * 0.70) 
training, test    = random_walk[0 : training_size], random_walk[training_size:]  

predictions       = list() 
hist              = training[-1] 

for i in range(len(test)): 
    yhat = hist 
    predictions.append(yhat) 
    hist = test[i] 
    
rmse = np.sqrt(mean_squared_error(test, predictions))  
print('\n\nPredicting a Random Walk with drift \nRMSE: %.3f' % rmse)
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
# Loading the data
bank_df = pd.read_csv('/kaggle/input/time-series-data/BOE-XUDLERD.csv')
bank_df
#converting to time series data 
bank_df['Date'] = pd.to_datetime(bank_df['Date'])
indexed_df = bank_df.set_index('Date')
ts = indexed_df['Value']
ts.head(5)
#Visualize the raw data

plt.plot(ts)
#Resample the data as it contains too much variations 

ts_week = ts.resample('W').mean()
plt.plot(ts_week)
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=52,center=False).mean() 
    rolstd = timeseries.rolling(window=52,center=False).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(ts_week)
ts_week_log = np.log(ts_week)
ts_week_log_diff = ts_week_log - ts_week_log.shift()
plt.plot(ts_week_log_diff)
#Again confirming with the dickey-fuller test 

ts_week_log_diff.dropna(inplace=True)
test_stationarity(ts_week_log_diff)
#ACF and PACF
lag_acf = acf(ts_week_log_diff, nlags=10)
lag_pacf = pacf(ts_week_log_diff, nlags=10, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-7.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
plt.axhline(y=7.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-7.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
plt.axhline(y=7.96/np.sqrt(len(ts_week_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
# Optimal values fot ARIMA(p,d,q) model are (2,1,1). Hence plot the ARIMA model using the value (2,1,1)
model = ARIMA(ts_week_log, order=(2, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_week_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_week_log_diff)**2))
# print the results of the ARIMA model and plot the residuals

print(results_ARIMA.summary())
# plot residual errors
residuals = DataFrame(results_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())
#Predictions 

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_week_log.iloc[0], index=ts_week_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts_week)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts_week)**2)/len(ts_week)))
#Training and testing datsets
size = int(len(ts_week_log) - 15)
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()
#Training the model and forecasting 

size = int(len(ts_week_log) - 15)
train, test = ts_week_log[0:size], ts_week_log[size:len(ts_week_log)]
history = [x for x in train]
predictions = list()
print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))
#Validating the model 

error = mean_squared_error(test, predictions)
print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)
predictions_series = pd.Series(predictions, index = test.index)
#Plotting forecasted vs Observed values 

fig, ax = plt.subplots()
ax.set(title='Spot Exchange Rate, Euro into USD', xlabel='Date', ylabel='Euro into USD')
ax.plot(ts_week[-60:], 'o', label='observed')
ax.plot(np.exp(predictions_series), 'g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
import sys
import warnings
import itertools
warnings.filterwarnings("ignore")

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
tractor_sales_Series = pd.read_csv("/kaggle/input/time-series-data/TractorSales.csv")
tractor_sales_Series.head(5)
dates = pd.date_range(start='2003-01-01', freq='MS', periods=len(tractor_sales_Series))
import calendar
data['Month'] = dates.month
data['Month'] = data['Month'].apply(lambda x: calendar.month_abbr[x])
data['Year'] = dates.year
#data.drop(['Month-Year'], axis=1, inplace=True)
data.rename(columns={'Number of Tractor Sold':'Tractor-Sales'}, inplace=True)
data = data[['Month', 'Year', 'Tractor-Sales']]
data.set_index(dates, inplace=True)
data.head(5)
# extract out the time-series
sales_ts = data['Tractor-Sales']
plt.figure(figsize=(8, 4))
plt.plot(sales_ts)
plt.xlabel('Years')
plt.ylabel('Tractor Sales')
fig, axes = plt.subplots(2, 2, sharey=False, sharex=False)
fig.set_figwidth(14)
fig.set_figheight(8)
axes[0][0].plot(sales_ts.index, sales_ts, label='Original')
axes[0][0].plot(sales_ts.index, sales_ts.rolling(window=4).mean(), label='4-Months Rolling Mean')
axes[0][0].set_xlabel("Years")
axes[0][0].set_ylabel("Number of Tractor's Sold")
axes[0][0].set_title("4-Months Moving Average")
axes[0][0].legend(loc='best')
axes[0][1].plot(sales_ts.index, sales_ts, label='Original')
axes[0][1].plot(sales_ts.index, sales_ts.rolling(window=6).mean(), label='6-Months Rolling Mean')
axes[0][1].set_xlabel("Years")
axes[0][1].set_ylabel("Number of Tractor's Sold")
axes[0][1].set_title("6-Months Moving Average")
axes[0][1].legend(loc='best')
axes[1][0].plot(sales_ts.index, sales_ts, label='Original')
axes[1][0].plot(sales_ts.index, sales_ts.rolling(window=8).mean(), label='8-Months Rolling Mean')
axes[1][0].set_xlabel("Years")
axes[1][0].set_ylabel("Number of Tractor's Sold")
axes[1][0].set_title("8-Months Moving Average")
axes[1][0].legend(loc='best')
axes[1][1].plot(sales_ts.index, sales_ts, label='Original')
axes[1][1].plot(sales_ts.index, sales_ts.rolling(window=12).mean(), label='12-Months Rolling Mean')
axes[1][1].set_xlabel("Years")
axes[1][1].set_ylabel("Number of Tractor's Sold")
axes[1][1].set_title("12-Months Moving Average")
axes[1][1].legend(loc='best')
plt.tight_layout()
plt.show()
#Determing rolling statistics

rolmean = sales_ts.rolling(window = 4).mean()
rolstd = sales_ts.rolling(window = 4).std()
#Plot rolling statistics:
orig = plt.plot(sales_ts, label='Original')
mean = plt.plot(rolmean, label='Rolling Mean')
std = plt.plot(rolstd, label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
from statsmodels.tsa.stattools import adfuller

dftest = adfuller(sales_ts)
dftest
print('DF test statistic is %3.3f' %dftest[0])
print('DF test p-value is %1.4f' %dftest[1])
monthly_sales_data = pd.pivot_table(data, values = "Tractor-Sales", columns = "Year", index = "Month")
monthly_sales_data
monthly_sales_data = monthly_sales_data.reindex(index = ['Jan','Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
monthly_sales_data
monthly_sales_data.plot()
yearly_sales_data = pd.pivot_table(data, values = "Tractor-Sales", columns = "Month", index = "Year")
yearly_sales_data = yearly_sales_data[['Jan','Feb','Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
yearly_sales_data
yearly_sales_data.plot()
yearly_sales_data.boxplot()
decomposition = sm.tsa.seasonal_decompose(sales_ts, model='multiplicative')
fig = decomposition.plot()
fig.set_figwidth(8)
fig.set_figheight(6)
fig.suptitle('Decomposition of multiplicative time series')
plt.show()
plt.figure(figsize=(8, 4))
plt.plot(sales_ts.diff(periods=1))
plt.xlabel('Years')
plt.ylabel('Tractor Sales')
plt.figure(figsize=(8, 4))
plt.plot(np.log10(sales_ts))
plt.xlabel('Years')
plt.ylabel('Log (Tractor Sales)')
plt.figure(figsize=(10, 5))
plt.plot(np.log10(sales_ts).diff(periods=1))
plt.xlabel('Years')
plt.ylabel('Differenced Log (Tractor Sales)')
sales_ts_log = np.log10(sales_ts)
sales_ts_log.dropna(inplace=True)

sales_ts_log_diff = sales_ts_log.diff(periods=1) # same as ts_log_diff = ts_log - ts_log.shift(periods=1)
sales_ts_log_diff.dropna(inplace=True)
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_acf(sales_ts_log, lags=30, ax=axes[0])
smt.graphics.plot_pacf(sales_ts_log, lags=30, ax=axes[1])
plt.tight_layout()
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(12)
fig.set_figheight(4)
plt.xticks(range(0,30,1), rotation = 90)
smt.graphics.plot_acf(sales_ts_log_diff, lags=30, ax=axes[0])
smt.graphics.plot_pacf(sales_ts_log_diff, lags=30, ax=axes[1])
plt.tight_layout()
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
pdq
seasonal_pdq
#Separate data into train and test
data['date'] = data.index
train = data[data.index < '2013-01-01']
test = data[data.index >= '2013-01-01']
train_sales_ts_log = np.log10(train['Tractor-Sales'])
best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        
        try:
            temp_model = sm.tsa.statespace.SARIMAX(train_sales_ts_log,
                                             order = param,
                                             seasonal_order = param_seasonal,
                                             enforce_stationarity=True)
            results = temp_model.fit()

            
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except:
            #print("Unexpected error:", sys.exc_info()[0])
            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
best_model = sm.tsa.statespace.SARIMAX(train_sales_ts_log,
                                      order=(0, 1, 1),
                                      seasonal_order=(1, 0, 1, 12),
                                      enforce_stationarity=True)
best_results = best_model.fit()
print(best_results.summary().tables[0])
print(best_results.summary().tables[1])
pred_dynamic = best_results.get_prediction(start=pd.to_datetime('2012-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
pred99 = best_results.get_forecast(steps=24, alpha=0.1)
# Extract the predicted and true values of our time series
sales_ts_forecasted = pred_dynamic.predicted_mean
testCopy = test.copy()
testCopy['sales_ts_forecasted'] = np.power(10, pred99.predicted_mean)
testCopy
# Compute the root mean square error
mse = ((testCopy['Tractor-Sales'] - testCopy['sales_ts_forecasted']) ** 2).mean()
rmse = np.sqrt(mse)
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 3)))
axis = train['Tractor-Sales'].plot(label='Train Sales', figsize=(10, 6))
testCopy['Tractor-Sales'].plot(ax=axis, label='Test Sales', alpha=0.7)
testCopy['sales_ts_forecasted'].plot(ax=axis, label='Forecasted Sales', alpha=0.7)
axis.set_xlabel('Years')
axis.set_ylabel('Tractor Sales')
plt.legend(loc='best')
plt.show()
plt.close()
# Get forecast 36 steps (3 years) ahead in future
n_steps = 36
pred_uc_99 = best_results.get_forecast(steps=36, alpha=0.01) # alpha=0.01 signifies 99% confidence interval
pred_uc_95 = best_results.get_forecast(steps=36, alpha=0.05) # alpha=0.05 95% CI

# Get confidence intervals 95% & 99% of the forecasts
pred_ci_99 = pred_uc_99.conf_int()
pred_ci_95 = pred_uc_95.conf_int()
n_steps = 36
idx = pd.date_range(data.index[-1], periods=n_steps, freq='MS')
fc_95 = pd.DataFrame(np.column_stack([np.power(10, pred_uc_95.predicted_mean), np.power(10, pred_ci_95)]), 
                     index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([np.power(10, pred_ci_99)]), 
                     index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
fc_all = fc_all[['forecast', 'lower_ci_95', 'upper_ci_95', 'lower_ci_99', 'upper_ci_99']] # just reordering columns
fc_all.head()
# plot the forecast along with the confidence band

axis = sales_ts.plot(label='Observed', figsize=(8, 4))
fc_all['forecast'].plot(ax=axis, label='Forecast', alpha=0.7)
axis.fill_between(fc_all.index, fc_all['lower_ci_95'], fc_all['upper_ci_95'], color='k', alpha=.15)
axis.set_xlabel('Years')
axis.set_ylabel('Tractor Sales')
plt.legend(loc='best')
plt.show()
best_results.plot_diagnostics(lags=30, figsize=(16,12))
plt.show()