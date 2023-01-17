import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

!pip install pmdarima

import pmdarima as pm

from pmdarima.model_selection import train_test_split

from pmdarima import arima

from pmdarima import model_selection

from pmdarima import pipeline

from pmdarima import preprocessing

from pmdarima.datasets._base import load_date_example

from pmdarima.utils import tsdisplay

from plotly.subplots import make_subplots

import plotly.graph_objects as go



# Load the data

mosquito_totals = pd.read_csv('../input/massachusetts-arbovirus-survalliance-data-201419/mosquito_totals.csv')



# Convert dates

mosquito_totals['Collection Date'] = pd.to_datetime(mosquito_totals['Collection Date'])



# Sort by Collection date

mosquito_totals = mosquito_totals.sort_values(by='Collection Date', ascending=True).reset_index()



print("Total number of observations:", len(mosquito_totals))



mosquito_totals.head()
# Print a random mosquito bite as a sample

sample_index = 25

print(mosquito_totals.iloc[sample_index])
import plotly.express as px

import plotly.graph_objects as go



mosquito_cases = pd.DataFrame(mosquito_totals.groupby(['Collection Date']).Virus.count()).reset_index()

mosquito_cases.columns = ['Date', 'Cases']



fig = go.Figure()

fig.add_trace(go.Scatter(x=mosquito_cases['Date'],

                         y=mosquito_cases['Cases'],

                         mode='lines',

                         name='Mosquito cases',

                         showlegend=True))

fig.update_layout(title='New confirmed cases per year',

                   xaxis_title='Year',

                   yaxis_title='New cases')

fig.show()
mosquito_totals['Year'] = mosquito_totals['Collection Date'].dt.year

mosquito_per_year = pd.DataFrame(mosquito_totals.groupby(['Year'])['Virus'].count()).reset_index()

mosquito_per_year.columns = ['Year', 'Cases']

fig = px.bar(mosquito_per_year, x='Year', y='Cases')

fig.update_layout(title='Total confirmed cases per year',

                   xaxis_title='Year',

                   yaxis_title='New cases')

fig.show()
mosquito_virus = mosquito_totals.groupby(['Year', 'Virus'], as_index=False).count()

mosquito_virus = mosquito_virus.iloc[:,:3]

mosquito_virus.columns = ['Year', 'Virus', 'Cases']

fig = px.bar(mosquito_virus, x="Year", y="Cases", color="Virus", title="Confirmed cases by virus")

fig.show()
mosquito_virus_sum = mosquito_virus.groupby(['Year', 'Virus'], as_index=False).agg('sum')

fig = px.line(mosquito_virus_sum, x="Year", y="Cases", color='Virus')

fig.update_layout(title='New EEE and WNV cases per year',

                   xaxis_title='Year',

                   yaxis_title='New cases')

fig.show()
# Remake the mosquito cases dataframe for clarity

mosquito_cases = pd.DataFrame(mosquito_totals.groupby(['Collection Date']).Virus.count()).reset_index()

mosquito_cases.columns = ['Date', 'Cases']



# Set date column to index

mosquito_cases.set_index('Date',inplace=True)



train_size = int(0.9*len(mosquito_cases))

y_train, y_test = train_test_split(mosquito_cases, train_size=train_size)



# Show the ACF and frequency plot of the data

tsdisplay(y_train, lag_max=90)
baseline_model = pm.auto_arima(y_train, suppress_warnings=True, start_p=0, start_q=0,

                      max_p=5, max_q=5, stepwise=True, trace=True, seasonal=True, m=12)
baseline_model.summary()
# Make predictions using naive model

predictions = baseline_model.predict(n_periods=y_test.shape[0])



# Plot baseline mode

fig = go.Figure()

x = np.arange(y_test.shape[0])

fig.add_trace(go.Scatter(x=x, y=y_test['Cases'], mode='markers', name='Actual cases', showlegend=True))

fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name='Predicated cases'))



fig.update_layout(title='Baseline naive model for m=12',

                   xaxis_title='Days',

                   yaxis_title='New cases')



fig.show()
from pmdarima.preprocessing import LogEndogTransformer



y_train_log, _ = LogEndogTransformer(lmbda=1e-6).fit_transform(y_train)

tsdisplay(y_train_log, lag_max=100)
from pmdarima.preprocessing import BoxCoxEndogTransformer



y_train_bc, _ = BoxCoxEndogTransformer(lmbda2=1e-6).fit_transform(y_train)

tsdisplay(y_train_bc, lag_max=100)
# Make a column containing the difference in cases for train and test set

# We shift by 12 since m=12



pd.set_option('mode.chained_assignment', None)

y_train['Cases Difference'] = y_train['Cases'] - y_train['Cases'].shift(12)

y_test['Cases Difference'] = y_test['Cases'] - y_test['Cases'].shift(12)



# Fill missing values (the first 12)

y_train = y_train.fillna(0)

y_test = y_test.fillna(0)



# Display the new data

pm.tsdisplay(y_train['Cases Difference'], lag_max=90, show=True)
x = np.arange(y_train.shape[0])



fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(x=x, y=y_train['Cases'], name="New cases"), row=1, col=1)

fig.add_trace(go.Scatter(x=x, y=y_train['Cases Difference'], name="New cases differentiated"), row=1, col=2)



fig.update_layout(title_text="Test samples count vs. differentiated")

fig.show()
# Create a default ARIMA model

model1 = pm.ARIMA(order=(1, 0, 1),

               seasonal_order=(1, 0, 1, 12),

               suppress_warnings=True)



# Create the one we found previously using AutoARIMA

model2 = pm.ARIMA(order=(2, 1, 1),

               seasonal_order=(0, 0, 1, 12),

               suppress_warnings=True)



# Set the CV strategy

cv = model_selection.SlidingWindowForecastCV(window_size=100, step=24, h=1)



# Run CV and get scores for each model

print("Creating model 1 (1, 0, 1): \n")

model1_cv_scores = model_selection.cross_val_score(model1, y_train['Cases Difference'], scoring='smape', cv=cv, verbose=2)

print()

print("Creating model 2 (2, 1, 1): \n")

model2_cv_scores = model_selection.cross_val_score(model2, y_train['Cases Difference'], scoring='smape', cv=cv, verbose=2)

print()



print("Model 1 CV scores: {}".format(model1_cv_scores.tolist()))

print("Model 2 CV scores: {}".format(model2_cv_scores.tolist()))



# Pick the with lowest mean error rate

m1_average_error = np.average(model1_cv_scores)

m2_average_error = np.average(model2_cv_scores)

errors = [m1_average_error, m2_average_error]

models = [model1, model2]



# Print out the answer

better_index = np.argmin(errors)  # type: int

print("Lowest average SMAPE: {} (model{})".format(errors[better_index], better_index + 1))

print("Best model: {}".format(models[better_index]))
from pmdarima import model_selection



# Create the model

model = pm.ARIMA(order=(2, 1, 1),

               seasonal_order=(0, 0, 1, 12),

               suppress_warnings=True)



# Set the CV strategy

cv = model_selection.SlidingWindowForecastCV(window_size=100, step=1, h=4)



# Make predictions

predictions = model_selection.cross_val_predict(model, y_train['Cases Difference'], cv=cv, verbose=1, averaging="median")



# Plot the predictions on the training data

x_axis = np.arange(y_train.shape[0])

n_test = predictions.shape[0]



fig = go.Figure()

fig.add_trace(go.Scatter(x=x_axis, y=y_train['Cases Difference'], name='Actual cases'))

fig.add_trace(go.Scatter(x=x_axis[-n_test:], y=predictions, name='Forecasted cases'))

fig.update_layout(title='Cross-validated mosquito forecasts', xaxis_title='Days', yaxis_title='New cases')

fig.show()
# Remove the first 12 zero samples

y_test = y_test.iloc[12:]



# Create the best model we found

model = pm.ARIMA(order=(2, 1, 1),

                 seasonal_order=(0, 0, 1, 12),

                 suppress_warnings=True)



# Fit on the difference score

model.fit(y_train['Cases Difference'])



# Make predictions

predictions = model.predict(n_periods=y_test.shape[0])



# Plot the result

x = np.arange(y_test.shape[0])

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y_test['Cases Difference'], mode='markers', name='Actual cases', showlegend=True))

fig.add_trace(go.Scatter(x=x, y=predictions, mode='lines', name='Forecasted cases'))



fig.update_layout(title='Forecasting 16 unseen mosquito cases',

                   xaxis_title='Days',

                   yaxis_title='New cases')

fig.show()
model.summary()
from sklearn.metrics import mean_squared_error as mse

from scipy.stats import normaltest



# Set axis

x = np.arange(y_train.shape[0] + predictions.shape[0])

fig, axes = plt.subplots(2, 1, sharex=False, figsize=(10,10))



# Make y_test an array

y_test_arr = np.array(y_test['Cases Difference'])



# Plot the forecasts

axes[0].plot(x[:y_train.shape[0]], y_train, c='b')

axes[0].plot(x[y_train.shape[0]:], predictions, c='g')

axes[0].set_xlabel(f'RMSE = {np.sqrt(mse(y_test_arr, predictions)):.3f}')

axes[0].set_xlabel('Days')

axes[0].set_ylabel('New cases +/-')

axes[0].set_title('Forecasting new mosquito cases')



# Plot the residuals

resid = y_test_arr - predictions

_, p = normaltest(resid)

axes[1].hist(resid, bins=15)

axes[1].axvline(0, linestyle='--', c='r')

axes[1].set_xlabel('New cases +/-')

axes[1].set_ylabel('Residaul strength')

axes[1].set_title(f'Residuals (p={p:.3f})')



plt.tight_layout()

plt.show()