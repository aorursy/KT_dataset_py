!pip install pmdarima

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pmdarima as pm

import numpy as np

import matplotlib.pyplot as plt
## Disclaimer I am not a trained professional this is a hobby prediction. Note that the model has high mean squared error.
global_data2 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

#global_data2 = pd.read_csv("../input/covid19-by-country-with-government-response/covid19_by_country.csv")

global_data2.tail()

#country_data = global_data2[global_data2['Country']=='Sweden']

country_data = global_data2[global_data2['Country/Region']=='Sweden']

#country_data = country_data[country_data['Date']>"2020-05-01"]

country_data = pd.pivot_table(country_data, values=['Confirmed', 'Recovered','Deaths'], index=['ObservationDate'], aggfunc=np.sum)

#country_data = pd.pivot_table(country_data, values=['Confirmed', 'Recovered','Deaths'], index=['ObservationDate'], aggfunc=np.sum)

#global_data2.tail()

country_data.tail()
from pandas.plotting import lag_plot



fig, axes = plt.subplots(5, 2, figsize=(12, 12))

plt.title('Confirmed Autocorrelation plot')



# The axis coordinates for the plots

ax_idcs = [

    (0, 0),

    (0, 1),

    (1, 0),

    (1, 1),

    (2, 0),

    (2, 1),

    (3, 0),

    (3, 1),

    (4, 0),

    (4, 1)

]



for lag, ax_coords in enumerate(ax_idcs, 1):

    ax_row, ax_col = ax_coords

    axis = axes[ax_row][ax_col]

    lag_plot(country_data['Confirmed'], lag=lag, ax=axis)

    axis.set_title(f"Lag={lag}")



plt.show()
from pmdarima.arima import ndiffs

from pmdarima.model_selection import train_test_split



# Train on Confirmed/Deaths

train_on='Confirmed'



# Can be changed between Confirmed and Deaths

train_len=country_data[train_on].size*0.8

y_train, y_test = train_test_split(country_data[train_on], train_size=int(train_len*0.8))



kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=30)

adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=30)

n_diffs = max(adf_diffs, kpss_diffs)

print(f"Estimated differencing term: {n_diffs}")
from pmdarima.utils import tsdisplay



tsdisplay(y_train, lag_max=130)
from pmdarima.preprocessing import LogEndogTransformer



y_train_log, _ = LogEndogTransformer(lmbda=1e-6).fit_transform(y_train)

tsdisplay(y_train_log, lag_max=130)
from pmdarima.preprocessing import BoxCoxEndogTransformer



y_train_bc, _ = BoxCoxEndogTransformer(lmbda2=1e-6).fit_transform(y_train)

tsdisplay(y_train_bc, lag_max=130)
auto = pm.auto_arima(y_train, d=n_diffs, seasonal=False, stepwise=True,

                     suppress_warnings=True, error_action="ignore", max_p=6,

                     max_order=None, trace=True)
print(auto.order)
auto.summary()
from sklearn.metrics import mean_squared_error

from pmdarima.metrics import smape



model = auto  # seeded from the model we've already fit



def forecast_one_step():

    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)

    return (

        fc.tolist()[0],

        np.asarray(conf_int).tolist()[0])



forecasts = []

confidence_intervals = []



for new_ob in y_test:

    fc, conf = forecast_one_step()

    forecasts.append(fc)

    confidence_intervals.append(conf)



    # Updates the existing model with a small number of MLE steps

    model.update(new_ob)



print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")

print(f"SMAPE: {smape(y_test, forecasts)}")
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# --------------------- Actual vs. Predicted --------------------------

axes[0].plot(y_train, color='blue', label='Training Data')

axes[0].plot(y_test.index, forecasts, color='green', marker='o',

             label='Predicted Cases')

axes[0].plot(y_test.index, y_test, color='red', label='Actual Cases')

axes[0].set_title('Swedid Covid Cases Prediction')

axes[0].set_xlabel('Dates')

axes[0].set_ylabel('Cases')

axes[0].legend()



axes[1].plot(y_train, color='blue', label='Training Data')

axes[1].plot(y_test.index, forecasts, color='green',

             label='Predicted Cases')

axes[1].set_title('Case Predictions & Confidence Intervals')

axes[1].set_xlabel('Dates')

axes[1].set_ylabel('Cases')

conf_int = np.asarray(confidence_intervals)

#axes[1].set_xticks(np.arange(0, 113, 90).tolist(), country_data['ObservationDate'][0:113:90].tolist())

axes[1].fill_between(y_test.index,

                     conf_int[:, 0], conf_int[:, 1],

                     alpha=0.9, color='orange',

                     label="Confidence Intervals")

axes[1].legend()
# Print predicted values ten periods from latest observationdate

predict=model.predict(n_periods=10)

np.around(predict, 0)
from sklearn.metrics import mean_squared_error as mse



#y_train.shape[0]

forecasts=model.predict(n_periods=10)

forecasts.shape[0]

x = np.arange(y_test.shape[0] + forecasts.shape[0])



fig, axes = plt.subplots(2, 1, sharex=False, figsize=(12,12))

# Plot the forecasts, this is predictions from the last observation date

axes[0].plot(x[:y_test.shape[0]], y_test, c='b')

axes[0].plot(x[y_test.shape[0]:], forecasts, c='g')