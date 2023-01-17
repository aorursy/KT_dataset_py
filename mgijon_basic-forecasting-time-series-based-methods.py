# Utilities 
import os
import datetime as dt
from datetime import timedelta

# Data manipulation
import numpy as np
import pandas as pd 

# Create plots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
%config InlineBackend.figure_format ='retina'   # To get high resolution plots

# To hide warnings
import warnings
warnings.filterwarnings('ignore');
# Load and preprocessing
data = pd.read_csv('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/cpsh.us.txt')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data.head()
# We have to define a date to separate training and test sets
trainig_set_size = 0.75

#plt.style.context('dark_background')
first_record = min(data.index)
last_record = max(data.index)
total_days = (last_record - first_record).days

# Total number of days
print('We have data from a total of ', total_days, ' days')   
training_days = np.ceil(total_days * trainig_set_size)
test_days = np.floor(total_days * (1 - trainig_set_size))

#date_split = first_record + int(total_test_days)
print('We are going to the data of ', int(training_days), ' days for training and ', int(test_days), ' days for testing.')
date_split = first_record + timedelta(days=training_days)
print('The train set goes from ' + str(first_record.year) + '-' + str(first_record.month) + '-' + str(first_record.day) + ' to ' + str(date_split.year) + '-' + str(date_split.month) + '-' + str(date_split.day))
print('The last day included in the training set is: ')

train_data = data[:date_split]
test_data = data[date_split:]

print(first_record)
print(last_record)
try: 
    if sum(data['OpenInt']) == 0:
        data = data.drop(columns=['OPE'], axis=1)
        pritn('The feature OpenInt has been deleted of the dataset because does not contain usefull information.')
except:
    pass
def plot_train_test(train, test, date_split, x_axis_title, y_axis_title):
    """Plot the time series, dividing it in train and test sets."""
    fig = go.Figure(data=[
                    go.Candlestick(
                            x=train.index,
                            open=train['Open'], 
                            high=train['High'],
                            low=train['Low'], 
                            close=train['Close'], 
                            name='Train',
                            increasing_line_color= 'cyan',
                            decreasing_line_color= 'gray'), 
                    go.Candlestick(
                            x=test.index,
                            open=test['Open'],
                            high=test['High'],
                            low=test['Low'],
                            close=test['Close'],
                            name='Test'), 
                    ])

    fig.update_layout(
        title=x_axis_title,
        yaxis_title=y_axis_title,
        shapes = [
                dict(x0=date_split, x1=test.index[-1], y0=0, y1=1, xref='x', yref='paper',line_width=2),
                dict(x0=train.index[0], x1=date_split, y0=0, y1=1, xref='x', yref='paper',line_width=2), ],
        annotations=[
                dict(x=date_split, y=0.05, xref='x', yref='paper', showarrow=False, xanchor='left', text='    Test set'),
                dict(x=date_split, y=0.05, xref='x', yref='paper', showarrow=False, xanchor='right', text='Train set    '), ]
    )

    fig.show()
    
plot_train_test(train=train_data, 
                test=test_data, 
                date_split=date_split, 
                x_axis_title='Stocks evolution, train and test sets', 
                y_axis_title='CPS Technologies Corporation')
def Average(y_hat, y):
    """Compute the Average of errors.
    :param y_hat: 1-Dimensional vector with predictions.
    :param y: 1-Dimensional vector with true values.
    :return average: Average.
    """
    errors = [y_hat[i] - y[i] for i in range(len(y))]
    T = len(errors)
    average = sum(errors) / T
    
    return average

def MAE(y_hat, y):
    """Compute the Mean Absolute Error (MAE).
    :param y_hat: 1-Dimensional vector with predictions.
    :param y: 1-Dimensional vector with true values.
    :return mae: Mean Absolute Error.
    """
    errors = [abs(y_hat[i] - y[i]) for i in range(len(y))]
    T = len(errors)
    mae = sum(errors) / T
    
    return mae 

def MSE(y_hat, y):
    """Compute the Mean Squared Error (MSR).
    :param y_hat: 1-Dimensional vector with predictions.
    :param y: 1-Dimensional vector with true values.
    :return msr: Mean Squared Error.
    """
    errors = [(y_hat[i] - y[i])**2 for i in range(len(y))]
    T = len(errors)
    msr = sum(errors) / T
    
    return msr

def RMSE(y_hat, y):
    """Compute the Root Mean Squared Error (RMSE).
    :param y_hat: 1-Dimensional vector with predictions.
    :param y: 1-Dimensional vector with true values.
    :return rmse: Root Mean Squared Error.
    """
    errors = [(y_hat[i] - y[i])**2 for i in range(len(y))]
    T = len(errors)
    rmse = np.sqrt(sum(errors) / T) 
    
    return rmse

def MAPE(y_hat, y):
    """Compute the Mean Absolute Percentage Error (MAPE).
    :param y_hat: 1-Dimensional vector with predictions.
    :param y: 1-Dimensional vector with true values.
    :return mape: Mean Absolute Percentage Error.
    """
    errors = [y_hat[i] - y[i] for i in range(len(y))]
    T = len(errors)
    numerator = sum( [abs(errors[i] / y[i]) for i in range(T)] )
    mape = 100 * (numerator / T)
    
    return mape

def MAPD(y_hat, y):
    """Compute Mean Absolute Percentage Deviation (MAPD).
    :param y_hat: 1-Dimensional vector with predictions.
    :param y: 1-Dimensional vector with true values.
    :param mapd: Mean Absolute Percentage Deviation.
    """
    errors = [y_hat[i] - y[i] for i in range(len(y))]
    numerator = sum( [abs(errors[i]) for i in range(len(errors))] )
    denominator = sum( [abs(y[i]) for i in range(len(y))] )
    mapd = numerator / denominator
    
    return mapd


results = {
    # Naive approach
    'Average': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'Naive': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'Drift': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    # Time series methods
    # Simple moving average for different values of window: 5, 10, 25, 50
    'SMA_5': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'SMA_10': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'SMA_25': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'SMA_50': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    # Cumulative moving average for different values of window: 5, 10, 25, 50
    'CMA_5': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'CMA_10': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'CMA_25': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'CMA_50': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    # Exponential moving average for different values of window: 5, 10, 25, 50
    'EMA_5': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'EMA_10': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'EMA_25': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
    'EMA_50': { 'average': 0, 'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mapd': 0 },
}

# hear a helper function
def fillErrors(algorithm, y_hat, y):
    """Compute the different types of errors for and save the results in the resutls variable.
    :param algorigthm: str, key for the results dictionary.
    :y_hat: 1-Dimensional vector of predictions.
    :y: 1-Dimensional vector of true values.
    """
    results[algorithm]['average'] = Average(y_hat, y)
    results[algorithm]['mae'] = MAE(y_hat, y)
    results[algorithm]['mse'] = MSE(y_hat, y)
    results[algorithm]['rmse'] = RMSE(y_hat, y)
    results[algorithm]['mape'] = MAPE(y_hat, y)
    results[algorithm]['mapd'] = MAPD(y_hat, y)
X = train_data['Close']
y = test_data['Close']

y_hat_AF = pd.Series([np.mean(X) for _ in range(len(y))], index=y.index)

                     
# Graphic
with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Average approach")
    plt.plot(X, label='Train values')
    plt.plot(y, label='Real value')
    # Predictions
    plt.plot(y_hat_AF, label='Average prediction', color='red')
    plt.plot(pd.Series([y_hat_AF.values[0] for _ in range(len(X))], index=X.index), color='red') # Extends to the previous values
    
    plt.legend()
    plt.show()

# Compute and save the results
fillErrors(algorithm='Average',  y_hat=y_hat_AF, y=y)
y_hat_NF = pd.Series([X[date_split] for _ in range(len(y))], index=y.index)
                     
# Graphic
with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Naive and Average approaches")
    plt.plot(X, label='Train values')
    plt.plot(y, label='Real value')
    # Predictions
    plt.plot(y_hat_AF, label='Average Forecast', color='red')
    plt.plot(pd.Series([y_hat_AF.values[0] for _ in range(len(X))], index=X.index), color='red') # Extends to the previous values
    plt.plot(y_hat_NF, label='Naive Forecast', color='green')
    plt.plot(pd.Series([y_hat_NF.values[0] for _ in range(len(X))], index=X.index), color='green') # Extends to the previous values
    
    plt.legend()
    plt.show()

# Compute the errors and save the resutls
fillErrors(algorithm='Naive',  y_hat=y_hat_NF, y=y)
y_t = X[-1]
m = (y_t - X[0]) / len(X)
h = np.linspace(0,len(y.index)-1, len(y.index))
y_hat_DF = pd.Series([y_t + m * h[i] for i in range(len(y.index))], index=y.index)

# Graphic
with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Naive, Average and Drift approaches")
    plt.plot(y, label='Real value')
    # Predictions
    plt.plot(y_hat_AF, label='Average Forecast', color='red')
    plt.plot(y_hat_NF, label='Naive Forecast', color='green')
    plt.plot(y_hat_DF, label='Drift Forecast', color='Yellow')
    plt.legend()
    plt.show()

# Compute the errors and save the resutls
fillErrors(algorithm='Drift', y_hat=y_hat_DF, y=y)
days = [5, 10, 25, 50]
colors = ['green', 'blue', 'yellow', 'purple']

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Simple Moving Average for different window size (train and test sets)")
    plt.plot(X, label='Train values', color='white')
    plt.plot(y, label='Real value')
    # Predictions
    for i in range(len(days)):
        SMA = data['Close'].rolling(window=days[i]).mean()
        plt.plot(SMA, label=days[i], color=colors[i])
        # Save the errors
        fillErrors(algorithm='SMA_' + str(days[i]), y_hat=y_hat_DF, y=y)
    plt.legend()
    plt.show()

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Simple Moving Average for different window size (test set)")
    plt.plot(y, label='Real value')
    # Predictions
    for i in range(len(days)):
        SMA = data['Close'].rolling(window=days[i]).mean()
        plt.plot(SMA[y.index], label=days[i], color=colors[i])

    plt.legend()
    plt.show()


days = [5, 10, 25, 50]
colors = ['green', 'blue', 'yellow', 'purple']

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Cumulative Moving Average for different window size (train and test set)")
    plt.plot(X, label='Train values', color='white')
    plt.plot(y, label='Real value')
    # Predictions
    for i in range(len(days)):
        SMA = data['Close'].expanding(min_periods=days[i]).mean()
        plt.plot(SMA, label=days[i], color=colors[i])
        # Save the errors
        fillErrors(algorithm='CMA_' + str(days[i]), y_hat=y_hat_DF, y=y)
    plt.legend()
    plt.show()

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Cumulative Moving Average for different window size (test set)")
    plt.plot(y, label='Real value')
    # Predictions
    for i in range(len(days)):
        SMA = data['Close'].expanding(min_periods=days[i]).mean()
        plt.plot(SMA[y.index], label=days[i], color=colors[i])

    plt.legend()
    plt.show()

days = [5, 10, 25, 50]
colors = ['green', 'blue', 'yellow', 'purple']

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Cumulative Moving Average for different window size (train and test sets)")
    plt.plot(X, label='Train values', color='white')
    plt.plot(y, label='Real value')
    # Predictions
    for i in range(len(days)):
        SMA = data['Close'].ewm(span=days[i],adjust=False).mean()
        plt.plot(SMA, label=days[i], color=colors[i])
        # Save the errors
        alg = 'EMA_' + str(days[i])
        fillErrors(algorithm=alg, y_hat=y_hat_DF, y=y)
    plt.legend()
    plt.show()

with plt.style.context('dark_background'):
    plt.figure(figsize=(20, 5.5))
    plt.title("Cumulative Moving Average for different window size (test set)")
    plt.plot(y, label='Real value')
    # Predictions
    for i in range(len(days)):
        SMA = data['Close'].ewm(span=days[i],adjust=False).mean()
        plt.plot(SMA[y.index], label=days[i], color=colors[i])

    plt.legend()
    plt.show()
error_types = []
methods = []
for key in results:
    methods.append(key)
for key in results[methods[0]]:
    error_types.append(key)
    
df = pd.DataFrame(index=methods, columns=error_types)
for method in methods:
    for error in error_types:
        df.loc[method, error] = results[method][error]
df
for error in error_types:
    plt.figure(figsize=(20, 5.5))
    plt.title('Error type: ' + error)
    df[error].plot(kind='bar', color=['green', 'blue', 'yellow', 'orange', 'red', 'brown', 'black', 'white',])
    plt.show()
