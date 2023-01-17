import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import timeit

start_time = timeit.default_timer()

train = pd.read_csv('../input/train_2.csv')

elapsed = timeit.default_timer() - start_time

print("Time to load data: ", round(elapsed, 2), "s")

print("Shape of Data: ", train.shape)
train.head()
train = train.transpose()

train.head(2)
new_header = train.iloc[0]

train = train[1:]

train.columns = new_header

train.head(3)
train.isna().sum().mean()
train = train.fillna(method = "ffill")

train.head(2)
train = train.fillna(method = "bfill")

train.head(2)
train.isna().sum().mean()
sample = train.sample(n = 100, axis = 1)

sample.head(2)
sample.shape
sample_test = sample.tail(30)

sample = sample.head(803 - 30)

sample.tail(2)
sample_test.head(2)
import warnings

warnings.filterwarnings("ignore")



# 1. Autoregression (AR)

from statsmodels.tsa.ar_model import AR



preds_data = pd.DataFrame()

start_date = "2017-08-12"

end_date = "2017-09-10"



# Measure time

start_time = timeit.default_timer()



# Fit model

for column in sample:

    model = AR(sample[column], freq = 'D')

    model_fit = model.fit()

# Make prediction

    yhat = model_fit.predict(start_date, end_date)

    preds_data[column] = yhat



# End time

elapsed = timeit.default_timer() - start_time



print("Time for 100 predictions: ", round(elapsed, 2), "s")

print("RMSE: ", (((preds_data - sample_test) ** 2).mean() ** 0.5).mean())

# 2. Moving Average (MA)

from statsmodels.tsa.arima_model import ARMA



preds_data = pd.DataFrame()

start_date = "2017-08-12"

end_date = "2017-09-10"



# Measure time

start_time = timeit.default_timer()



# Fit model

for column in sample:

    model = ARMA(sample[column], order = (0,1), freq = 'D')

    model_fit = model.fit()

# Make prediction

    yhat = model_fit.predict(start_date, end_date)

    preds_data[column] = yhat



# End time

elapsed = timeit.default_timer() - start_time



print("Time for 100 predictions: ", round(elapsed, 2), "s")

print("RMSE: ", (((preds_data - sample_test) ** 2).mean() ** 0.5).mean())
import warnings

warnings.filterwarnings("ignore")



# 3. Autoregressive Moving Average (ARMA)

from statsmodels.tsa.arima_model import ARMA



preds_data = pd.DataFrame()

start_date = "2017-08-12"

end_date = "2017-09-10"



# Measure time

start_time = timeit.default_timer()



# Fit model

for column in sample:

    model = ARMA(sample[column], order = (1,0), freq = 'D')

    model_fit = model.fit()

# Make prediction

    yhat = model_fit.predict(start_date, end_date)

    preds_data[column] = yhat



# End time

elapsed = timeit.default_timer() - start_time



print("Time for 100 predictions: ", round(elapsed, 2), "s")

print("RMSE: ", (((preds_data - sample_test) ** 2).mean() ** 0.5).mean())
# 4. Autoregressive Integrated Moving Average (ARIMA)

from statsmodels.tsa.arima_model import ARIMA



preds_data = pd.DataFrame()

start_date = "2017-08-12"

end_date = "2017-09-10"



# Measure time

start_time = timeit.default_timer()



# Fit model

for column in sample:

    model = ARIMA(sample[column], order = (1, 0, 0), freq = 'D')

    model_fit = model.fit()

# Make prediction

    yhat = model_fit.predict(start_date, end_date)

    preds_data[column] = yhat



# End time

elapsed = timeit.default_timer() - start_time



print("Time for 100 predictions: ", round(elapsed, 2), "s")

print("RMSE: ", (((preds_data - sample_test) ** 2).mean() ** 0.5).mean())
# 5. SARIMAX

from statsmodels.tsa.statespace.sarimax import SARIMAX



preds_data = pd.DataFrame()

start_date = "2017-08-12"

end_date = "2017-09-10"



# Measure time

start_time = timeit.default_timer()



# Fit model

for column in sample:

    model = SARIMAX(sample[column], freq = 'D')

    model_fit = model.fit()

# Make prediction

    yhat = model_fit.forecast(steps = 30)

    preds_data[column] = yhat



# End time

elapsed = timeit.default_timer() - start_time



print("Time for 100 predictions: ", round(elapsed, 2), "s")

print("RMSE: ", (((preds_data - sample_test) ** 2).mean() ** 0.5).mean())
# 6. SARIMAX parameters

from statsmodels.tsa.statespace.sarimax import SARIMAX



preds_data = pd.DataFrame()

start_date = "2017-08-12"

end_date = "2017-09-10"



# Measure time

start_time = timeit.default_timer()



# Fit model

for column in sample:

    model = SARIMAX(sample[column], order = (1,1,0), freq = 'D')

    model_fit = model.fit()

# Make prediction

    yhat = model_fit.forecast(steps = 30)

    preds_data[column] = yhat



# End time

elapsed = timeit.default_timer() - start_time



print("Time for 100 predictions: ", round(elapsed, 2), "s")

print("RMSE: ", (((preds_data - sample_test) ** 2).mean() ** 0.5).mean())
# 7. Simple Exponential Smoothing (SES)

from statsmodels.tsa.holtwinters import SimpleExpSmoothing



preds_data = pd.DataFrame()

start_date = "2017-08-12"

end_date = "2017-09-10"



# Measure time

start_time = timeit.default_timer()



# Fit model

for column in sample:

    model = SimpleExpSmoothing(sample[column])

    model_fit = model.fit()

# Make prediction

    yhat = model_fit.predict(start_date, end_date)

    preds_data[column] = yhat



# End time

elapsed = timeit.default_timer() - start_time



print("Time for 100 predictions: ", round(elapsed, 2), "s")

print("RMSE: ", (((preds_data - sample_test) ** 2).mean() ** 0.5).mean())
# 8. Holt Winters Exponential Smoothing (HWES)

from statsmodels.tsa.holtwinters import ExponentialSmoothing



preds_data = pd.DataFrame()

start_date = "2017-08-12"

end_date = "2017-09-10"



# Measure time

start_time = timeit.default_timer()



# Fit model

for column in sample:

    model = ExponentialSmoothing(sample[column])

    model_fit = model.fit()

# Make prediction

    yhat = model_fit.predict(start_date, end_date)

    preds_data[column] = yhat



# End time

elapsed = timeit.default_timer() - start_time



print("Time for 100 predictions: ", round(elapsed, 2), "s")

print("RMSE: ", (((preds_data - sample_test) ** 2).mean() ** 0.5).mean())