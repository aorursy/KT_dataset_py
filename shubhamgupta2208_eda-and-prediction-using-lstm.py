import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format
sns.set(rc={'figure.figsize':(20, 20)})
import sys
print("Python version: {}". format(sys.version))

import pandas as pd 
print("pandas version: {}". format(pd.__version__))

import matplotlib 
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np 
print("NumPy version: {}". format(np.__version__))

import scipy as sp 
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display 
print("IPython version: {}". format(IPython.__version__)) 

import sklearn 
print("scikit-learn version: {}". format(sklearn.__version__))

import keras
print("keras version: {}".format(keras.__version__))

import tensorflow as tf
print("tensorflow version: {}".format(tf.__version__))
df = pd.read_csv('../input/all_stocks_2006-01-01_to_2018-01-01.csv')
df.head()
df = pd.read_csv('../input/all_stocks_2006-01-01_to_2018-01-01.csv', parse_dates=['Date'])
df.info()
df.Date = pd.to_datetime(df.Date)
df.describe()
df.isnull().sum()
df[df.Open.isnull()]
rng = pd.date_range(start='2006-01-01', end='2018-01-01', freq='B')
rng[~rng.isin(df.Date.unique())]
df.groupby('Name').count().sort_values('Date', ascending=False)['Date']
gdf = df[df.Name == 'AABA']
cdf = df[df.Name == 'CAT']
cdf[~cdf.Date.isin(gdf.Date)]
# Total number of companies
df.Name.unique().size
df.groupby('Date').Name.unique().apply(len)
df.set_index('Date', inplace=True)

#Backfill `Open` column
values = np.where(df['2017-07-31']['Open'].isnull(), df['2017-07-28']['Open'], df['2017-07-31']['Open'])
df['2017-07-31']= df['2017-07-31'].assign(Open=values.tolist())

values = np.where(df['2017-07-31']['Close'].isnull(), df['2017-07-28']['Close'], df['2017-07-31']['Close'])
df['2017-07-31']= df['2017-07-31'].assign(Close=values.tolist())

values = np.where(df['2017-07-31']['High'].isnull(), df['2017-07-28']['High'], df['2017-07-31']['High'])
df['2017-07-31']= df['2017-07-31'].assign(High=values.tolist())

values = np.where(df['2017-07-31']['Low'].isnull(), df['2017-07-28']['Low'], df['2017-07-31']['Low'])
df['2017-07-31']= df['2017-07-31'].assign(Low=values.tolist())

df.reset_index(inplace=True)

df[df.Date == '2017-07-31']
missing_data_stocks = ['CSCO','AMZN','INTC','AAPL','MSFT','MRK','GOOGL', 'AABA']
columns = df.columns.values
for stock in missing_data_stocks:
    tdf = df[(df.Name == stock) & (df.Date == '2014-03-28')].copy()
    tdf.Date = '2014-04-01'
    pd.concat([df, tdf])
print("Complete")
df[(df.Name == 'CSCO') & (df.Date == '2014-04-01')]
df[df.Open.isnull()]
df = df[~((df.Date == '2012-08-01') & (df.Name == 'DIS'))]
df.isnull().sum()
values = (df['High'] + df['Low'] + df['Open'] + df['Close'])/4
df = df.assign(Price=values)
df.head()
df.Price.describe()
stock_names = df.Name.unique()
day_prices = df[df.Date == df.Date.min()].Price
price_mapping = {n : c for n, c in zip(stock_names, day_prices)}
base_mapping = np.array(list(map(lambda x : price_mapping[x], df['Name'].values)))
df['Growth'] = df['Price'] / base_mapping - 1
df.Growth.describe()
sample_dates = pd.date_range(start='2006-01-01', end='2018-01-01', freq='B')
year_end_dates = sample_dates[sample_dates.is_year_end]
year_end_dates
worst_stocks = df[df.Date == df.Date.max()].sort_values('Growth').head(5)
best_stocks = df[df.Date == df.Date.max()].sort_values('Growth', ascending=False).head(5)
ws = worst_stocks.Name.values
bs = best_stocks.Name.values
tdf = df.copy()
tdf = df.set_index('Date')
tdf[tdf.Name.isin(ws)].groupby('Name').Growth.plot(title='Historical trend of worst 5 stocks of 2017', legend=True)
tdf[tdf.Name.isin(bs)].groupby('Name').Growth.plot(title='Historical trend of best 5 stocks of 2017', legend=True)
worst_stocks
best_stocks
corr = df.pivot('Date', 'Name', 'Growth').corr()
sns.heatmap(corr)
def unique_corelations(indices):
    mapping = {}
    for record in indices:
        (stock_a, stock_b) = record
        value_list = mapping.get(stock_a)
        if value_list:
            if stock_b not in value_list:
                value_list.append(stock_b)
                mapping.update({stock_a: value_list})
        else:
            mapping.update({stock_a: [stock_b]})

    return mapping

def filter_corelations_positive(corr, threshold=0.9):
    indices = np.where(corr > threshold)
    indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
    mapping = unique_corelations(indices)
    return mapping
    
def filter_corelations_negative(corr, threshold=-0.8):
    indices = np.where(corr < threshold)
    indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
    mapping = unique_corelations(indices)
    return mapping
filter_corelations_positive(corr, threshold=0.95)
filter_corelations_negative(corr, -0.1)
google_df = df[df.Name == 'GOOGL']
gdf = google_df[['Date', 'Price']].sort_values('Date')
training_set = gdf[gdf.Date.dt.year != 2017].Price.values
test_set =  gdf[gdf.Date.dt.year == 2017].Price.values
print("Training set size: ",training_set.size)
print("Test set size: ", test_set.size)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
training_set_scaled = scaler.fit_transform(training_set.reshape(-1, 1))
def create_train_data(training_set_scaled):
    X_train, y_train = [], []
    for i in range(30, training_set_scaled.size):
        X_train.append(training_set_scaled[i-30: i])
        y_train.append(training_set_scaled[i])
    # Converting list to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train
X_train, y_train = create_train_data(training_set_scaled)
def create_test_data():
    X_test = []
    inputs = gdf[len(gdf) - len(test_set) - 30:].Price.values
    inputs = scaler.transform(inputs.reshape(-1, 1))
    for i in range(30, test_set.size+30): # Range of the number of values in the training dataset
        X_test.append(inputs[i - 30: i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test
X_test = create_test_data()
X_test.shape
from keras.models import Sequential
from keras.layers import Dense, LSTM
def create_simple_model():
    model = Sequential()
    model.add(LSTM(units = 10, return_sequences = False, input_shape = (X_train.shape[1], 1)))
    model.add(Dense(units = 1))
    return model
def compile_and_run(model, epochs=50, batch_size=64):
    model.compile(metrics=['accuracy'], optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=3)
    return history
def plot_metrics(history):
    metrics_df = pd.DataFrame(data={"loss": history.history['loss']})
    metrics_df.plot()
simple_model = create_simple_model()
history = compile_and_run(simple_model, epochs=20)
plot_metrics(history)
def make_predictions(X_test, model):
    y_pred = model.predict(X_test)
    final_predictions = scaler.inverse_transform(y_pred)
    fp = np.ndarray.flatten(final_predictions)
    ap = np.ndarray.flatten(test_set)
    pdf = pd.DataFrame(data={'Actual': ap, 'Predicted': fp})
    ax = pdf.plot()
make_predictions(X_test, simple_model)
def create_dl_model():
    model = Sequential()

    # Adding the first LSTM layer
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

    # Adding a second LSTM layer
    model.add(LSTM(units = 50, return_sequences = True))
    
    # Adding a third LSTM layer
    model.add(LSTM(units = 50, return_sequences = True))

    # Adding a fourth LSTM layer
    model.add(LSTM(units = 50))

    # Adding the output layer
    model.add(Dense(units = 1))
    return model
dl_model = create_dl_model()
dl_model.summary()
history = compile_and_run(dl_model, epochs=20)
plot_metrics(history)
make_predictions(X_test, dl_model)