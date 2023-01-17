!pip install nsepy
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import nsepy as ns

from datetime import date
infy = ns.get_history(symbol='INFY',start=date(2015,4,1), end=date(2016,3,31))    #.to_pickle('./infy.pkl')

tcs = ns.get_history(symbol='TCS',start=date(2015,4,1), end=date(2016,3,31))      #.to_pickle('./tcs.pkl')
display(infy.head())

display(tcs.head())
infy['Close'].plot()
tcs['Close'].plot()
def plot_moving_average(series, window, plot_intervals=False, scale=1.96):



    rolling_mean = series.rolling(window=window).mean()

    

    plt.figure(figsize=(15,6))

    plt.title('Moving average\n window size = {}'.format(window))

    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    

    #Plot confidence intervals for smoothed values

    if plot_intervals:

        mae = mean_absolute_error(series[window:], rolling_mean[window:])

        deviation = np.std(series[window:] - rolling_mean[window:])

        lower_bound = rolling_mean - (mae + scale * deviation)

        upper_bound = rolling_mean + (mae + scale * deviation)

        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')

        plt.plot(lower_bound, 'r--')

            

    plt.plot(series[window:], label='Actual values')

    plt.legend(loc='best')

    plt.grid(True)
plot_moving_average(infy['Close'],10) 

plot_moving_average(infy['Close'],75)
plot_moving_average(tcs['Close'],10)

plot_moving_average(tcs['Close'],75)
infy.head()
infy['Vol_per_incr'] = np.zeros(infy.shape[0])

infy['ClosePr_per_incr'] = np.zeros(infy.shape[0])
for i in range(infy.shape[0]-1):

    infy['Vol_per_incr'].iloc[i+1] = ((infy['Volume'].iloc[i+1] - infy['Volume'].iloc[i])/infy['Volume'].iloc[i]) *100

    infy['ClosePr_per_incr'].iloc[i+1] = ((infy['Close'].iloc[i+1] - infy['Close'].iloc[i])/infy['Close'].iloc[i]) *100
infy.head()
infy['dummy_Vol_incr'] = pd.get_dummies(abs(infy['Vol_per_incr']) > 10, drop_first=True)

infy['dummy_Close_incr'] = pd.get_dummies(abs(infy['ClosePr_per_incr']) > 2,drop_first=True)

infy.head()
tcs['Vol_per_incr'] = np.zeros(tcs.shape[0])

tcs['ClosePr_per_incr'] = np.zeros(tcs.shape[0])



for i in range(tcs.shape[0]-1):

    tcs['Vol_per_incr'].iloc[i+1] = ((tcs['Volume'].iloc[i+1] - tcs['Volume'].iloc[i])/tcs['Volume'].iloc[i]) *100

    tcs['ClosePr_per_incr'].iloc[i+1] = ((tcs['Close'].iloc[i+1] - tcs['Close'].iloc[i])/tcs['Close'].iloc[i]) *100
tcs['dummy_Vol_incr'] = pd.get_dummies(abs(tcs['Vol_per_incr']) > 10, drop_first=True)

tcs['dummy_Close_incr'] = pd.get_dummies(abs(tcs['ClosePr_per_incr']) > 2,drop_first=True)

tcs.head()
#Price without Volume



infy['Pr_without_Vol'] = np.zeros(infy.shape[0])

tcs['Pr_without_Vol'] = np.zeros(tcs.shape[0])



for i in range(infy.shape[0]):

    if infy['dummy_Vol_incr'].iloc[i] == 1 and infy['dummy_Close_incr'].iloc[i] == 0:

        infy['Pr_without_Vol'].iloc[i] = 1

    else:

        infy['Pr_without_Vol'].iloc[i] = 0

        



for i in range(tcs.shape[0]-1):

    if tcs['dummy_Vol_incr'].iloc[i] == 1 and tcs['dummy_Close_incr'].iloc[i] == 0:

        tcs['Pr_without_Vol'].iloc[i] = 1

    else:

        tcs['Pr_without_Vol'].iloc[i] = 0
infy.head()
from bokeh.plotting import figure, show, output_file

from bokeh.models import ColumnDataSource

from bokeh.io import output_notebook
source = ColumnDataSource(infy)

source_tcs = ColumnDataSource(tcs)



p = figure(x_axis_type='datetime', plot_width=800, plot_height=350)

p.line('Date','Close', source=source, line_color='blue')

p.line('Date','Close', source=source_tcs, line_color='red')



output_notebook()

show(p)
p=figure(x_axis_label = 'Volume', y_axis_label = 'Date')

p.line('Date','Volume',source=source, line_color='blue')

p.line('Date','Volume', source=source_tcs, line_color='red')



output_notebook()

show(p)
# Plotting auto correlation 



from statsmodels.tsa.stattools import pacf



lag_pacf = pacf(infy['Close'], nlags=20, method='ols')

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='red')

plt.axhline(y=-1.96/np.sqrt(len(infy['Close'])),linestyle='--',color='red')

plt.axhline(y=1.96/np.sqrt(len(infy['Close'])),linestyle='--',color='red')

plt.title('Partial_Autocorrelation Function')
lag_pacf = pacf(tcs['Close'], nlags=20, method='ols')

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='red')

plt.axhline(y=-1.96/np.sqrt(len(tcs['Close'])),linestyle='--',color='red')

plt.axhline(y=1.96/np.sqrt(len(tcs['Close'])),linestyle='--',color='red')

plt.title('Partial_Autocorrelation Function')
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold

from sklearn.metrics import mean_squared_error,r2_score 
df_infy = infy.drop(columns=['Symbol','Series','VWAP','Turnover','Trades','Deliverable Volume','%Deliverble',

                            'Vol_per_incr','ClosePr_per_incr','dummy_Vol_incr','dummy_Close_incr'])



df_tcs = tcs.drop(columns=['Symbol','Series','VWAP','Turnover','Trades','Deliverable Volume','%Deliverble',

                            'Vol_per_incr','ClosePr_per_incr','dummy_Vol_incr','dummy_Close_incr'])
df_infy.head()
X_train,X_test,y_train,y_test = train_test_split(df_infy.drop(columns=['Close']),df_infy['Close'],test_size= 0.25, shuffle= True)
lr = LinearRegression()
scores = cross_val_score(lr, X_train, y_train, cv = 5)    #cv is the number of folds, scores will give an array of scores

print(scores) 

print(np.mean(scores))

print(np.std(scores))
predictions = cross_val_predict(lr, X_test, y_test, cv = 5)

r2_score(y_test, predictions)
lr.fit(X_train,y_train)

pred = lr.predict(X_test)
print(lr.score(X_test,y_test))

print(mean_squared_error(y_test,pred))

print(r2_score(y_test,pred))
random_forest = RandomForestRegressor(n_jobs=-1)
scores = cross_val_score(random_forest, X_train, y_train, cv = 5)

print(scores) 

print(np.mean(scores))

print(np.std(scores))
predictions = cross_val_predict(random_forest, X_test, y_test, cv = 5)

r2_score(y_test, predictions)
random_forest.fit(X_train,y_train)

pred = random_forest.predict(X_test)
print(mean_squared_error(y_test,pred))

print(r2_score(y_test,pred))