import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, GlobalMaxPooling1D, Bidirectional
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

%matplotlib inline

# supress annoying warning
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
df_confirmed = pd.read_csv("../input/covid-19/time_series_covid19_confirmed_global.csv")
df_deaths = pd.read_csv("../input/covid-19/time_series_covid19_deaths_global.csv")
df_reco = pd.read_csv("../input/covid-19/time_series_covid19_recovered_global.csv")
df_confirmed.head()
df_deaths.head()
df_reco.head()
us_confirmed = df_confirmed[df_confirmed["Country/Region"] == "US"]
us_deaths = df_deaths[df_deaths["Country/Region"] == "US"]
us_reco = df_reco[df_reco["Country/Region"] == "US"]

germany_confirmed = df_confirmed[df_confirmed["Country/Region"] == "Germany"]
germany_deaths = df_deaths[df_deaths["Country/Region"] == "Germany"]
germany_reco = df_reco[df_reco["Country/Region"] == "Germany"]

italy_confirmed = df_confirmed[df_confirmed["Country/Region"] == "Italy"]
italy_deaths = df_deaths[df_deaths["Country/Region"] == "Italy"]
italy_reco = df_reco[df_reco["Country/Region"] == "Italy"]

sk_confirmed = df_confirmed[df_confirmed["Country/Region"] == "Korea, South"]
sk_deaths = df_deaths[df_deaths["Country/Region"] == "Korea, South"]
sk_reco = df_reco[df_reco["Country/Region"] == "Korea, South"]

us_reco
## structuring timeseries data
def confirmed_timeseries(df):
    df_series = pd.DataFrame(df[df.columns[4:]].sum(),columns=["confirmed"])
    df_series.index = pd.to_datetime(df_series.index,format = '%m/%d/%y')
    return df_series

def deaths_timeseries(df):
    df_series = pd.DataFrame(df[df.columns[4:]].sum(),columns=["deaths"])
    df_series.index = pd.to_datetime(df_series.index,format = '%m/%d/%y')
    return df_series

def reco_timeseries(df):
    # no index to timeseries conversion needed (all is joined later)
    df_series = pd.DataFrame(df[df.columns[4:]].sum(),columns=["recovered"])
    return df_series
us_con_series = confirmed_timeseries(us_confirmed)
us_dea_series = deaths_timeseries(us_deaths)
us_reco_series = reco_timeseries(us_reco)

germany_con_series = confirmed_timeseries(germany_confirmed)
germany_dea_series = deaths_timeseries(germany_deaths)
germany_reco_series = reco_timeseries(germany_reco)

italy_con_series = confirmed_timeseries(italy_confirmed)
italy_dea_series = deaths_timeseries(italy_deaths)
italy_reco_series = reco_timeseries(italy_reco)

sk_con_series = confirmed_timeseries(sk_confirmed)
sk_dea_series = deaths_timeseries(sk_deaths)
sk_reco_series = reco_timeseries(sk_reco)
# join all data frames for each county (makes it easier to graph and compare)

us_df = us_con_series.join(us_dea_series, how = "inner")
us_df = us_df.join(us_reco_series, how = "inner")

germany_df = germany_con_series.join(germany_dea_series, how = "inner")
germany_df = germany_df.join(germany_reco_series, how = "inner")

italy_df = italy_con_series.join(italy_dea_series, how = "inner")
italy_df = italy_df.join(italy_reco_series, how = "inner")

sk_df = sk_con_series.join(sk_dea_series, how = "inner")
sk_df = sk_df.join(sk_reco_series, how = "inner")
us_df
us_df.plot(figsize=(14,7),title="United States confirmed, deaths and recoverd cases")
us_cases_outcome = (us_df.tail(1)['deaths'] + us_df.tail(1)['recovered'])[0]
us_outcome_perc = (us_cases_outcome / us_df.tail(1)['confirmed'] * 100)[0]
us_death_perc = (us_df.tail(1)['deaths'] / us_cases_outcome * 100)[0]
us_reco_perc = (us_df.tail(1)['recovered'] / us_cases_outcome * 100)[0]
us_active = (us_df.tail(1)['confirmed'] - us_cases_outcome)[0]

print(f"Number of cases which had an outcome: {us_cases_outcome}")
print(f"percentage of cases that had an outcome: {round(us_outcome_perc, 2)}%")
print(f"Deaths rate: {round(us_death_perc, 2)}%")
print(f"Recovery rate: {round(us_reco_perc, 2)}%")
print(f"Currently Active cases: {us_active}")

germany_df.plot(figsize=(14,7),title="Germany confirmed, deaths and recoverd cases")
germany_cases_outcome = (germany_df.tail(1)['deaths'] + germany_df.tail(1)['recovered'])[0]
germany_outcome_perc = (germany_cases_outcome / germany_df.tail(1)['confirmed'] * 100)[0]
germany_death_perc = (germany_df.tail(1)['deaths'] / germany_cases_outcome * 100)[0]
germany_reco_perc = (germany_df.tail(1)['recovered'] / germany_cases_outcome * 100)[0]
germany_active = (germany_df.tail(1)['confirmed'] - germany_cases_outcome)[0]

print(f"Number of cases which had an outcome: {germany_cases_outcome}")
print(f"percentage of cases that had an outcome: {round(germany_outcome_perc, 2)}%")
print(f"Deaths rate: {round(germany_death_perc, 2)}%")
print(f"Recovery rate: {round(germany_reco_perc, 2)}%")
print(f"Currently Active cases: {germany_active}")

italy_df.plot(figsize=(14,7),title="Italy confirmed, deaths and recoverd cases")
italy_cases_outcome = (italy_df.tail(1)['deaths'] + italy_df.tail(1)['recovered'])[0]
italy_outcome_perc = (italy_cases_outcome / italy_df.tail(1)['confirmed'] * 100)[0]
italy_death_perc = (italy_df.tail(1)['deaths'] / italy_cases_outcome * 100)[0]
italy_reco_perc = (italy_df.tail(1)['recovered'] / italy_cases_outcome * 100)[0]
italy_active = (italy_df.tail(1)['confirmed'] - italy_cases_outcome)[0]

print(f"Number of cases which had an outcome: {italy_cases_outcome}")
print(f"percentage of cases that had an outcome: {round(italy_outcome_perc, 2)}%")
print(f"Deaths rate: {round(italy_death_perc, 2)}%")
print(f"Recovery rate: {round(italy_reco_perc, 2)}%")
print(f"Currently Active cases: {italy_active}")

sk_df.plot(figsize=(14,7),title="South Korea confirmed, deaths and recoverd cases")
sk_cases_outcome = (sk_df.tail(1)['deaths'] + sk_df.tail(1)['recovered'])[0]
sk_outcome_perc = (sk_cases_outcome / sk_df.tail(1)['confirmed'] * 100)[0]
sk_death_perc = (sk_df.tail(1)['deaths'] / sk_cases_outcome * 100)[0]
sk_reco_perc = (sk_df.tail(1)['recovered'] / sk_cases_outcome * 100)[0]
sk_active = (sk_df.tail(1)['confirmed'] - sk_cases_outcome)[0]

print(f"Number of cases which had an outcome: {sk_cases_outcome}")
print(f"percentage of cases that had an outcome: {round(sk_outcome_perc, 2)}%")
print(f"Deaths rate: {round(sk_death_perc, 2)}%")
print(f"Recovery rate: {round(sk_reco_perc, 2)}%")
print(f"Currently Active cases: {sk_active}")
n_input = 7  # number of steps
n_features = 1 # number of y 

# prepare required input data
def prepare_data(df):
    # drop rows with zeros
    df = df[(df.T != 0).any()]
    
    num_days = len(df) - n_input
    train = df.iloc[:num_days]
    test = df.iloc[num_days:]
    
    # normalize the data according to largest value
    scaler = MinMaxScaler()
    scaler.fit(train) # find max value

    scaled_train = scaler.transform(train) # divide every point by max value
    scaled_test = scaler.transform(test)
    
    # feed in batches [t1,t2,t3] --> t4 
    generator = TimeseriesGenerator(scaled_train,scaled_train,length = n_input,batch_size = 1)
    validation_set = np.append(scaled_train[55],scaled_test) # random tbh
    validation_set = validation_set.reshape(n_input + 1,1)
    validation_gen = TimeseriesGenerator(validation_set,validation_set,length = n_input,batch_size = 1)
    
    return scaler, train, test, scaled_train, scaled_test, generator, validation_gen
    
# create, train and return LSTM model
def train_lstm_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(84, recurrent_dropout = 0, unroll = False, return_sequences = True, use_bias = True, input_shape = (n_input,n_features))))
    model.add(LSTM(84, recurrent_dropout = 0.1, use_bias = True, return_sequences = True,))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(84, activation = "relu"))
    model.add(Dense(units = 1))
    
    # compile model
    model.compile(loss = 'mae', optimizer = Adam(1e-5))
    
    # finally train the model using generators
    model.fit_generator(generator,validation_data = validation_gen, epochs = 100, steps_per_epoch = round(len(train) / n_input), verbose = 0)
    
    return model
# predict, rescale and append needed columns to final data frame
def lstm_predict(model):
    # holding predictions
    test_prediction = []

    # last n points from training set
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape(1,n_input,n_features)
    
    # predict first x days from testing data 
    for i in range(len(test) + n_input):
        current_pred = model.predict(current_batch)[0]
        test_prediction.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    # inverse scaled data
    true_prediction = scaler.inverse_transform(test_prediction)

    MAPE, accuracy, sum_errs, interval, stdev, df_forecast = gen_metrics(true_prediction)

    return MAPE, accuracy, sum_errs, interval, stdev, df_forecast

# plotting model losses
def plot_lstm_losses(model):
    pd.DataFrame(model.history.history).plot(figsize = (14,7), title = "loss vs epochs curve")
'''
incrementally trained ARIMA:
    - train with original train data
    - predict the next value
    - appened the prediction value to the training data
    - repeat training and appending for n times (days in this case)
    
    this incremental technique significantly improves the accuracy
    by always using all data up to previous day for predeicting next value
    unlike predecting multiple values at the same time which is not incremeital 
'''

def arima_predict():
    values = [x for x in train.values]
    predictions = []
    for t in range(len(test) + n_input): # the number of testing days + the future days to predict 
        model = ARIMA(values, order = (7,1,1))
        model_fit = model.fit()
        fcast = model_fit.forecast()
        predictions.append(fcast[0][0])
        values.append(fcast[0])
    
    MAPE, accuracy, sum_errs, interval, stdev, df_forecast = gen_metrics(predictions)
    
    return MAPE, accuracy, sum_errs, interval, stdev, df_forecast
'''
incremental Holt's (Method) Exponential Smoothing
    - trained the same way as above arima
'''
def hes_predict():   
    values = [x for x in train.values]
    predictions = []
    for t in range(len(test) + n_input): # the number of testing days + the future days to predict 
        model = Holt(values)
        model_fit = model.fit()
        fcast = model_fit.predict()
        predictions.append(fcast[0])
        values.append(fcast[0])
        
    MAPE, accuracy, sum_errs, interval, stdev, df_forecast = gen_metrics(predictions)
    
    return MAPE, accuracy, sum_errs, interval, stdev, df_forecast
# generate metrics and final df
def gen_metrics(pred):
    # create time series
    time_series_array = test.index
    for k in range(0, n_input):
        time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

    # create time series data frame
    df_forecast = pd.DataFrame(columns = ["confirmed","confirmed_predicted"],index = time_series_array)
    
    # append confirmed and predicted confirmed
    df_forecast.loc[:,"confirmed_predicted"] = pred
    df_forecast.loc[:,"confirmed"] = test["confirmed"]
    
    # create and append daily cases (for both actual and predicted)
    daily_act = []
    daily_pred = []
    
    #actual
    daily_act.append(df_forecast["confirmed"].iloc[1] - train["confirmed"].iloc[-1])
    for num in range((n_input * 2) - 1):
        daily_act.append(df_forecast["confirmed"].iloc[num + 1] - df_forecast["confirmed"].iloc[num])
    
    # predicted
    daily_pred.append(df_forecast["confirmed_predicted"].iloc[1] - train["confirmed"].iloc[-1])
    for num in range((n_input * 2) - 1):
        daily_pred.append(df_forecast["confirmed_predicted"].iloc[num + 1] - df_forecast["confirmed_predicted"].iloc[num])
    
    df_forecast["daily"] = daily_act
    df_forecast["daily_predicted"] = daily_pred
    
    # calculate mean absolute percentage error
    MAPE = np.mean(np.abs(np.array(df_forecast["confirmed"][:n_input]) - np.array(df_forecast["confirmed_predicted"][:n_input])) / np.array(df_forecast["confirmed"][:n_input]))

    accuracy = round((1 - MAPE) * 100, 2)

    # the error rate
    sum_errs = np.sum((np.array(df_forecast["confirmed"][:n_input]) - np.array(df_forecast["confirmed_predicted"][:n_input])) ** 2)

    # error standard deviation
    stdev = np.sqrt(1 / (n_input - 2) * sum_errs)

    # calculate prediction interval
    interval = 1.96 * stdev

    # append the min and max cases to final df
    df_forecast["confirm_min"] = df_forecast["confirmed_predicted"] - interval
    df_forecast["confirm_max"] = df_forecast["confirmed_predicted"] + interval
    
    # round all df values to 0 decimal points
    df_forecast = df_forecast.round() 
    
    return MAPE, accuracy, sum_errs, interval, stdev, df_forecast
# print metrics for given county
def print_metrics(mape, accuracy, errs, interval, std, model_type):
    m_str = "LSTM" if model_type == 0 else "ARIMA" if model_type == 1 else "HES"
    print(f"{m_str} MAPE: {round(mape * 100, 2)}%")
    print(f"{m_str} accuracy: {accuracy}%")
    print(f"{m_str} sum of errors: {round(errs)}")
    print(f"{m_str} prediction interval: {round(interval)}")
    print(f"{m_str} standard deviation: {std}")
# for plotting the range of predicetions
def plot_results(df, country, algo):
    fig, (ax1, ax2) = plt.subplots(2, figsize = (14,20))
    ax1.set_title(f"{country} {algo} confirmed predictions")
    ax1.plot(df.index,df["confirmed"], label = "confirmed")
    ax1.plot(df.index,df["confirmed_predicted"], label = "confirmed_predicted")
    ax1.fill_between(df.index,df["confirm_min"], df["confirm_max"], color = "indigo",alpha = 0.09,label = "Confidence Interval")
    ax1.legend(loc = 2)
    
    ax2.set_title(f"{country} {algo} confirmed daily predictions")
    ax2.plot(df.index, df["daily"], label = "daily")
    ax2.plot(df.index, df["daily_predicted"], label = "daily_predicted")
    ax2.legend()
    
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    fig.show()
# prepare the data

scaler, train, test, scaled_train, scaled_test, generator, validation_gen = prepare_data(us_con_series)
# train lstm model
us_lstm_model = train_lstm_model()

# plot lstm losses
plot_lstm_losses(us_lstm_model)
# Long short memory method
us_mape, us_accuracy, us_errs, us_interval, us_std, us_lstm_df = lstm_predict(us_lstm_model)

print_metrics(us_mape, us_accuracy, us_errs, us_interval, us_std, 0)

us_lstm_df
plot_results(us_lstm_df, "USA", "LSTM")
# Auto Regressive Integrated Moving Average 

us_mape, us_accuracy, us_errs, us_interval, us_std, us_arima_df = arima_predict()

print_metrics(us_mape, us_accuracy, us_errs, us_interval, us_std, 1)

us_arima_df
plot_results(us_arima_df, "USA", "incremental ARIMA")
# Holts Exponential Smoothing
us_mape, us_accuracy, us_errs, us_interval, us_std, us_hes_df = hes_predict()

print_metrics(us_mape, us_accuracy, us_errs, us_interval, us_std, 2)

us_hes_df
plot_results(us_hes_df, "USA", "incremental HES")
# prepare the data
scaler, train, test, scaled_train, scaled_test, generator, validation_gen = prepare_data(germany_con_series)
# train the model
germany_lstm_model = train_lstm_model()

# plot losses
plot_lstm_losses(germany_lstm_model)
# LSTM
germany_mape, germany_accuracy, germany_errs, germany_interval, germany_std, germany_lstm_df = lstm_predict(germany_lstm_model)

print_metrics(germany_mape, germany_accuracy, germany_errs, germany_interval, germany_std, 0)

germany_lstm_df
plot_results(germany_lstm_df, "Germany", "LSTM")
# ARIMA
germany_mape, germany_accuracy, germany_errs, germany_interval, germany_std, germany_arima_df = arima_predict()

print_metrics(germany_mape, germany_accuracy, germany_errs, germany_interval, germany_std, 1)

germany_arima_df
plot_results(germany_arima_df, "Germany", "incremental ARIMA")
# HES
germany_mape, germany_accuracy, germany_errs, germany_interval, germany_std, germany_hes_df = hes_predict()

print_metrics(germany_mape, germany_accuracy, germany_errs, germany_interval, germany_std, 1)

germany_hes_df
plot_results(germany_hes_df, "Germany", "incremental HES")
# prepare the data
scaler, train, test, scaled_train, scaled_test, generator, validation_gen = prepare_data(italy_con_series)
# train the model
italy_lstm_model = train_lstm_model()

# plot lstm losses
plot_lstm_losses(italy_lstm_model)
# LSTM
italy_mape, italy_accuracy, italy_errs, italy_interval, italy_std, italy_lstm_df = lstm_predict(italy_lstm_model)

print_metrics(italy_mape, italy_accuracy, italy_errs, italy_interval, italy_std, 0)

italy_lstm_df
plot_results(italy_lstm_df, "Italy", "LSTM")
# ARIMA
italy_mape, italy_accuracy, italy_errs, italy_interval, italy_std, italy_arima_df = arima_predict()

print_metrics(italy_mape, italy_accuracy, italy_errs, italy_interval, italy_std, 1)

italy_arima_df
plot_results(italy_arima_df, "Italy", "incremental ARIMA")
# HES
italy_mape, italy_accuracy, italy_errs, italy_interval, italy_std, italy_hes_df = hes_predict()

print_metrics(italy_mape, italy_accuracy, italy_errs, italy_interval, italy_std, 2)

italy_hes_df
plot_results(italy_hes_df, "Italy", "incremental HES")
# prepare the data
scaler, train, test, scaled_train, scaled_test, generator, validation_gen = prepare_data(sk_con_series)
# train the model
sk_lstm_model = train_lstm_model()

# plot lstm losses
plot_lstm_losses(sk_lstm_model)
# get important metrics
sk_mape, sk_accuracy, sk_errs, sk_interval, sk_std, sk_lstm_df = lstm_predict(sk_lstm_model)

print_metrics(sk_mape, sk_accuracy, sk_errs, sk_interval, sk_std, 0)

sk_lstm_df
plot_results(sk_lstm_df, "South Korea", "LSTM")
sk_mape, sk_accuracy, sk_errs, sk_interval, sk_std, sk_arima_df = arima_predict()

print_metrics(sk_mape, sk_accuracy, sk_errs, sk_interval, sk_std, 1)


sk_arima_df
plot_results(sk_arima_df, "South Korea", "incremental ARIMA")
sk_mape, sk_accuracy, sk_errs, sk_interval, sk_std, sk_hes_df = hes_predict()

print_metrics(sk_mape, sk_accuracy, sk_errs, sk_interval, sk_std, 1)


sk_hes_df
plot_results(sk_hes_df, "South Korea", "incremental HES")