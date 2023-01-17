# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Read Data "TSLA.csv" and set "Date" as INDEX for the dataset
data = pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv", index_col ="Date", parse_dates = True)
data.shape
data.head()
# Plot "Adj Close" and "Volume" of Tesla Shares
plt.figure(figsize=(20, 15))

plt.subplot(2,1,1)
plt.plot(data['Adj Close'], label='Adj Close', color="purple")
plt.legend(loc="upper right")
plt.title('Adj Close Prices of Tesla')

plt.subplot(2,1,2)
plt.plot(data['Volume'], label='Volume', color="Orange")
plt.legend(loc="upper right")
plt.title('Volume Of Shares Traded')
# To install the library 
!pip install pmdarima 
# Import the library 
from pmdarima import auto_arima 
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 
  
# Fit auto_arima function to AirPassengers dataset 
stepwise_fit = auto_arima(data['Adj Close'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',    # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 
  
# To print the summary 
stepwise_fit.summary() 
# Split data into train / test sets 
train = data.iloc[:len(data)-12] 
test = data.iloc[len(data)-12:] # set one year(12 months) for testing 
  
# Fit a SARIMAX(1, 0, 2)x(0, 1, [1], 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['Adj Close'],  
                order = (1, 0, 2),  
                seasonal_order =(0, 1, 1, 12)) 
  
result = model.fit() 
result.summary() 
start = len(train) 
end = len(train) + len(test) - 1
  
# Predictions for one-year against the test set 
predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# Create dataframe of Predictions
predictions_df = pd.DataFrame(predictions)
predictions_df.index = ['2020-01-16', '2020-01-17', '2020-01-21', '2020-01-22',
               '2020-01-23', '2020-01-24', '2020-01-27', '2020-01-28',
               '2020-01-29', '2020-01-30', '2020-01-31', '2020-02-03']
predictions_df.index = pd.to_datetime(predictions_df.index)

# plot predictions and actual values 
predictions_df.plot(legend = True) 
test['Adj Close'].plot(legend = True)
# Load specific evaluation tools 
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 
  
# Calculate root mean squared error 
print("RMSE on Test Data: ", rmse(test["Adj Close"], predictions))
  
# Calculate mean squared error 
print("MSE on Test Data: ", mean_squared_error(test["Adj Close"], predictions))

# Train the model on the full dataset 
model = model = SARIMAX(data['Adj Close'],  
                        order = (1, 0, 2),  
                        seasonal_order =(0, 1, 1, 12)) 
    
result = model.fit() 
  
# Forecast for the next 1 Month 
forecast = result.predict(start = len(data),  
                          end = (len(data)-1) + 1,             # +1 means 1 month advance from the last date i.e. 2nd Feb 2020
                          typ = 'levels').rename('Forecast') 

print("The predicted share price on the 3rd March 2020 is: {}".format(forecast.iloc[0]))
