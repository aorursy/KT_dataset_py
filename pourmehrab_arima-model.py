import pandas as pd
df = pd.read_csv('time_series.csv', index_col='date', parse_dates=True)
# Creating ARMA data
from statsmodels.tsa.arima_process import arma_generate_sample
ar_coefs = [1, -0.9, -0.1]
ma_coefs = [1, 0.2]
y = pd.DataFrame(arma_generate_sample(ar_coefs, ma_coefs, nsample=1000, sigma=0.5))
from matplotlib import pyplot
y.plot(style='k.')
pyplot.show()
from statsmodels.tsa.stattools import adfuller
adf = adfuller(y)
print('ADF Statistic:', adf[0])
print('p-value:', adf[1])
df_stationary = y.diff().dropna()
# Fitting and ARMA model
from statsmodels.tsa.arima_model import ARMA
# Instantiate model object
model = ARMA(y, order=(1,1)) # order(p,q)
# Fit model
results = model.fit()
print(results.summary())
# Fitting ARMAX
# Instantiate the model
model = ARMA(df['productivity'], order=(2,1), exog=df['hours_sleep'])
# Fit the model
results = model.fit()
# Statsmodels SARIMAX class
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Just an ARMA(p,q) model
model = SARIMAX(df, order=(p,0,q))
# An ARMA(p,q) + constant model
model = SARIMAX(df, order=(p,0,q), trend='c')
# Make predictions for last 25 values
results = model.fit()
# Make in-sample prediction
forecast = results.get_prediction(start=-25) # ,dynamic=True)
# forecast mean
mean_forecast = forecast.predicted_mean
# Get confidence intervals of forecasts
confidence_intervals = forecast.conf_int()
# Forecasting out of sample
forecast = results.get_forecast(steps=20)
# Assign residuals to variable
residuals = results.resid
mae = np.mean(np.abs(residuals))
# Create the 4 diagostics plots
results.plot_diagnostics()
plt.show()
# Plotting predictions
plt.figure()
# Plot prediction
plt.plot(dates,
mean_forecast.values,
color='red',
label='forecast')
# Shade uncertainty area
plt.fill_between(dates, lower_limits, upper_limits, color='pink')
plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
# Make ACF plot
plot_acf(df, lags=10, zero=False, ax=ax1)
# Make PACF plot
plot_pacf(df, lags=10, zero=False, ax=ax2)
plt.show()
# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# Decompose data
decomp_results = seasonal_decompose(df['IPG3113N'], freq=12)
# Plot decomposed data
decomp_results.plot()
plt.show()
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Instantiate model
model = SARIMAX(df, order=(p,d,q), seasonal_order=(P,D,Q,S))
# Fit model
results = model.fit()
# Searching over model orders
import pmdarima as pm

# Non-seasonal search parameters
results = pm.auto_arima( df, # data
d=0, # non-seasonal difference order
start_p=1, # initial guess for p
start_q=1, # initial guess for q
max_p=3, # max value of p to test
max_q=3, # max value of q to test
)

# Seasonal search parameters
results = pm.auto_arima( df, # data
seasonal=True, # is the time series seasonal
m=7, # the seasonal period
D=1, # seasonal difference order
start_P=1, # initial guess for P
start_Q=1, # initial guess for Q
max_P=2, # max value of P to test
max_Q=2, # max value of Q to test
)

# results = pm.auto_arima( df, # data
# ... , # model order parameters
# information_criterion='aic', # used to select best model
# trace=True, # print results whilst training
# error_action='ignore', # ignore orders that don't work
# stepwise=True, # apply intelligent order search
# )

print(results.summary())
results.plot_diagnostics()
# Saving model objects
import joblib
# Select a filepath
filepath ='localpath/great_model.pkl'
# Save model to filepath
joblib.dump(model_results_object, filepath)
# Load model object from filepath
model_results_object = joblib.load(filepath)
# Add new observations and update parameters
model_results_object.update(df_new)