import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import matplotlib.pylab as plt
import tensorflow as tf
from sklearn import preprocessing
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
items.tail()
categories = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
categories.tail()
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
shops
sales = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
sales.tail()
sales_test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
sales_test.tail()
print(sum(items.duplicated(['item_name'])))
print(sum(categories.duplicated(['item_category_name'])))
print(sum(shops.duplicated(['shop_name'])))
# We can see that the names of shops 10 and 11 differ only by one letter. It is probably the same shop.  
# Also 0 and 57, 1 and 58, ?39 and 40?
# Let's find out if shops 10,11,0,57,1,58 present in the dataframe for forecasting.
uniq_shops = sales_test['shop_id'].unique()
for shop in list([10,11,0,57,1,58]):
    print(shop, shop in uniq_shops)
new_shop_id = {11: 10, 0: 57, 1: 58}
shops['shop_id'] = shops['shop_id'].apply(lambda x: new_shop_id[x] if x in new_shop_id.keys() else x)
sales['shop_id'] = sales['shop_id'].apply(lambda x: new_shop_id[x] if x in new_shop_id.keys() else x)
sales = pd.merge(sales_test, sales, on = ('shop_id', 'item_id'), how = 'left')
print(sum(sales.duplicated()))
# drop the duplicate rows from sales
sales = sales.drop_duplicates()
sales.shape
print(sum(sales.duplicated(['ID','date','date_block_num','item_price'])))
print(sum(sales.duplicated(['ID','date','date_block_num','item_cnt_day'])))
# We should think carefully which row should be dropped. Price will help us. But now we will just keep first duplicate and drop later.
sales = sales.drop_duplicates(['date','date_block_num','shop_id','item_id','item_cnt_day'])
sales.shape
sales = sales.drop_duplicates(['ID','date','date_block_num'], keep = 'last')
sales.shape
print(items.isnull().sum().sum())
print(categories.isnull().sum().sum())
print(shops.isnull().sum().sum())
print(sales.isnull().sum().sum())
# There are missing values in the data. Most of them corresponds to IDs from the forcast set that doesn't represent in training set.
sales.describe()
# It is possible that item_price and item_cnt_day has outliers (max >> 0.75-quantile), and item_cnt_day has wrong values (min < 0)
# change a sign of negative values
sales.loc[sales.item_cnt_day < 0, 'item_cnt_day'] = -1. * sales.loc[sales.item_cnt_day < 0, 'item_cnt_day']
#sales_month = sales.sort_values('date_block_num').groupby(['ID', 'date_block_num'], as_index = False).agg({'item_cnt_day': ['sum'], 'item_price': ['mean']})
#sales_month.columns = ['ID', 'date_block_num', 'item_cnt_month', 'item_price']
sales_month = sales.sort_values('date_block_num').groupby(['ID', 'date_block_num'], as_index = False).agg({'item_cnt_day': ['sum']})
sales_month.columns = ['ID', 'date_block_num', 'item_cnt_month']
sales_month.sample(10)
# after we grouped and aggregate data we delete all rows corresponding to IDs that don't present in train data set (and preset just in forcasting set)
sales_month.describe()
def to_IDs(np_data, col_ID):
    # np_data - sales converted to numpy array
    # col_ID - name of ID column
    sales_by_ID = list()
    IDs = np.unique(np_data[:,col_ID]).astype(int)
    for i in IDs:
        positions = np_data[:,col_ID] == i
        sales_ID = np_data[positions,1:]
        sales_by_ID.append(sales_ID)
    return sales_by_ID, IDs
sales_by_ID, list_IDs = to_IDs(sales_month.values,0)
print(len(sales_by_ID))
# to decrease calculation time during a code debugging we remove IDs that don't have observtions for last months
def remove_ID_nan_last_year(np_data):
    N_IDs = len(np_data)
    col_date = 0
    clear_data = list()
    cut_month = 33 - 2
    for i in range(N_IDs):
        ID_data = np_data[i]
        if len(ID_data[ID_data[:,col_date] >= cut_month,1]) != 0:
            clear_data.append(ID_data)
    return clear_data
#sales_by_ID = remove_ID_nan_last_year(sales_by_ID)
#len(sales_by_ID)
#val_month = 28
#train, test_actual = split_train_test(sales_by_ID, last_month = val_month)
#test_actual = np.nan_to_num(test_actual, nan = 0)
# Let's fill the missing date_block_num by NaN for paticular ID
def missing_months(np_data, col_date, col_TS, N_months = 34):
    # col_date - index of date_block_num column
    # col_TS - index of item_price column and item_cnt_month column
    # at first fill time series by NaN for all months
    series = [np.nan for _ in range(N_months)]
    for i in range(len(np_data)):
        position = int(np_data[i, col_date] - 1)
        # fill positions that present in data
        series[position] = np_data[i, col_TS]
    return series
# Let's fill the missing item_cnt_month and item_price for particular ID
def to_fill_missing(np_data, N_months = 34):
    col = ['date_block_num','item_cnt_month']
    sales_ID = pd.DataFrame(np_data, columns = col)
    if sales_ID.shape[0] < N_months:
        date_month = pd.DataFrame(range(N_months),columns = ['date_block_num'])
        sales_ID = pd.merge(date_month, sales_ID, on = ('date_block_num'), how = 'left')
        sales_ID = sales_ID.reindex(columns = col)
        sales_ID['item_cnt_month'] = sales_ID['item_cnt_month'].fillna(0.0)
    return sales_ID['item_cnt_month'].to_numpy()
# Plot time series for particular ID to find out missing months
def plot_TS(np_data, n_vars = 1, N_months = 34, flag = 0):
    # n_vars = 1 or 2 (plot item_cnt OR item_cnt and item_price)
    plt.figure()
    if flag == 1:
        TSs = to_fill_missing(np_data, N_months)
    for i in range(n_vars):
        col_plot = i + 1 # index of column to plot
        if flag == 1:
            series = TSs#[:,col_plot]
        else:
            series = missing_months(np_data, 0, col_plot, N_months)
        ax = plt.subplot(n_vars, 1, i+1)
        plt.plot(series, 'o')
        plt.plot(series)
    plt.show()
for i in np.random.randint(0, len(sales_by_ID), 5):
    plot_TS(sales_by_ID[i], flag = 1)
# Let's create 2D-array and each column is counts of particular ID where missing months is filled
def full_data(data, N_months = 34):
    N_IDs = len(data)
    TS = np.empty((N_months, N_IDs))
    for i in range(N_IDs):
        TS[:, i] = to_fill_missing(data[i], N_months)
    return TS
TS = full_data(sales_by_ID)
TS.shape
val_month = 29
valid_TS = TS[val_month:,:]
print(valid_TS.shape)

train_TS = TS[:val_month,:]
scaler = preprocessing.MinMaxScaler()
scaler.fit(TS)
train_scaled = scaler.transform(train_TS)
valid_scaled = scaler.transform(valid_TS)
def to_make_features(TS, n_lag, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(TS) # each element of dataset is one value of TS 
    ds = ds.window(n_lag+1, shift = 1, drop_remainder = True) # (n_lag+1)-elements of dataset is combined to window
    ds = ds.flat_map(lambda row: row.batch(n_lag + 1)) # to batch elements in window to tensor (one element) and to flat (now there are no windows)
  # Let's shuffle befor we combine batches for epoch
    ds = ds.shuffle(300)
  # make the tuple: first element is features, second element is labels
  # features-(1,2,3) and labels-(2,3,4). 2 goes after 1, 3 goes after 2, 4 goes after 3.
    ds = ds.map(lambda row: (row[:-1,:], row[1:,:]))
  # combine tuples to banch for gradient descent
  # instead of a row we will have a matrix in every tuple
    ds = ds.batch(batch_size).prefetch(1)
    return ds
import random
tf.random.set_seed(53)
random.seed(53)
n_lag = 6
batch_size = 8
features = to_make_features(train_scaled, n_lag, batch_size)
val_features = to_make_features(valid_scaled, n_lag, batch_size)
Conv_filters = 64
Conv_kernel_size = 4
LSTM_filters = 64
n_outputs = train_scaled.shape[1]
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters = Conv_filters, kernel_size = Conv_kernel_size,
                      strides=1, padding="causal", activation="relu", input_shape=[None, n_outputs]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
  tf.keras.layers.Dense(n_outputs)
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5)

model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

model.summary()
fitting = model.fit(features, epochs=80, callbacks=[lr_schedule])
plt.semilogx(fitting.history["lr"], fitting.history["loss"])
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
fitting = model.fit(features, epochs=300, verbose = 1, validation_data = val_features)
mae = fitting.history['mae']
loss = fitting.history['loss']
epochs=range(len(loss))
plt.plot(epochs, mae, 'r')
plt.plot(epochs, fitting.history['val_mae'], 'r--')
plt.plot(epochs, loss, 'b')
plt.plot(epochs, fitting.history['val_loss'], 'b--')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "val_MAE", "Loss", "val_Loss"])
plt.plot(epochs, fitting.history['val_mae'], 'r--')
def model_forecast(model, TS, n_lag, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(TS)
    ds = ds.window(n_lag, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda row: row.batch(n_lag))
    ds = ds.batch(batch_size)
    forecast = model.predict(ds)
    return forecast
forecast = model_forecast(model, train_scaled, n_lag, batch_size)
forecast = forecast[:,-1,:]
forecast = scaler.inverse_transform(forecast)
iplot = 0
for i in np.random.randint(0, n_outputs, 4):
    iplot += 1
    plt.subplot(4,1,iplot)
    plt.plot(range(n_lag, val_month+1), np.append(train_TS[n_lag:,i],valid_TS[0,i]), 'r')
    plt.plot(range(n_lag, val_month+1), forecast[:,i], 'b')
    plt.legend(["actual", "predicted"])
lag_set = range(2,34-val_month,2)
batch_size_set = np.array([4,8,16])
mae_val = np.zeros((len(lag_set), len(batch_size_set)))
mae_train = np.zeros((len(lag_set), len(batch_size_set)))
i, j = 0, 0
for batch_size in batch_size_set:
    for lag in lag_set:
        features = to_make_features(train_scaled, lag, batch_size)
        val_features = to_make_features(valid_scaled, lag, batch_size)
        model = tf.keras.models.Sequential([
              tf.keras.layers.Conv1D(filters = Conv_filters, kernel_size = Conv_kernel_size,
                                  strides=1, padding="causal", activation="relu", input_shape=[None, n_outputs]),
              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
              tf.keras.layers.Dense(n_outputs)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
        
        fitting = model.fit(features, epochs = 100, verbose = 1, validation_data = val_features)
        mae_val_i = np.array(fitting.history['val_mae'])
        mae_val[i,j] = np.min(mae_val_i[np.nonzero(mae_val_i)])
        min_position = fitting.history['val_mae'].index(mae_val[i,j])
        mae_train[i,j] = fitting.history['mae'][min_position]
        i += 1
    j += 1
    i = 0
plt.subplot(2,1,1)
for j in range(len(batch_size_set)):
    plt.plot(lag_set, mae_train[:,j])
plt.legend(["batch 1", "batch 2", "batch 3"])
plt.title("MAE")
plt.subplot(2,1,2)
for j in range(len(batch_size_set)):
    plt.plot(lag_set, mae_val[:,j])
plt.legend(["batch 1", "batch 2", "batch 3"])
plt.title("val_MAE")
# 4 lags and 16 batch size are the best
n_lag = 4
batch_size = 16
features = to_make_features(train_scaled, n_lag, batch_size)
val_features = to_make_features(valid_scaled, n_lag, batch_size)
epoch_set = np.array([50, 100, 300, 500, 1000])
mae_val_ep = np.zeros(len(epoch_set))
mae_train_ep = np.zeros(len(epoch_set))
i = 0
for epoch in epoch_set:
    fitting = model.fit(features, epochs = epoch)
    forecast_val = model_forecast(model, train_scaled[33-n_lag:,:], n_lag, batch_size)
    forecast_val = forecast_val[:,-1,:]
    forecast_val = scaler.inverse_transform(forecast_val)
    mae_val_ep[i] = np.mean(np.abs(forecast_val - test_actual))
    mae_train_ep[i] = fitting.history['mae'][-1]
    i += 1
plt.subplot(2,1,1)
plt.plot(epoch_set, mae_train_ep)
plt.subplot(2,1,2)
plt.plot(epoch_set, mae_val_ep)
# 500 epochs is the best
n_epoch = 100
Conv_filters_set = [32, 64, 256]
Conv_kernel_size_set = [2, 4, 6]
LSTM_filters_set = [32, 64, 256, 512]
mae_val = np.zeros((len(Conv_filters_set), len(Conv_kernel_size_set)))
mae_train = np.zeros((len(Conv_filters_set), len(Conv_kernel_size_set)))
i, j = 0, 0
for Conv_filters in Conv_filters_set:
    for Conv_kernel_size in Conv_kernel_size_set:
        model = tf.keras.models.Sequential([
          tf.keras.layers.Conv1D(filters = Conv_filters, kernel_size = Conv_kernel_size,
                      strides=1, padding="causal", activation="relu", input_shape=[None, n_outputs]),
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
          tf.keras.layers.Dense(n_outputs)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
        fitting = model.fit(features, epochs = n_epoch, verbose = 1, validation_data = val_features)

        mae_val_i = np.array(fitting.history['val_mae'])
        mae_val[i,j] = np.min(mae_val_i[np.nonzero(mae_val_i)])
        min_position = fitting.history['val_mae'].index(mae_val[i,j])
        mae_train[i,j] = fitting.history['mae'][min_position]
        i += 1
    j += 1
    i = 0
plt.subplot(2,1,1)
for j in range(len(Conv_filters_set)):
    plt.plot(Conv_kernel_size_set, mae_train[:,j])
plt.legend(["Filter 1", "Filter 2", "Filter 3"])
plt.subplot(2,1,2)
for j in range(len(Conv_filters_set)):
    plt.plot(Conv_kernel_size_set, mae_val[:,j])
plt.legend(["Filter 1", "Filter 2", "Filter 3"])
mae_train
mae_val
Conv_filters = 32
Conv_kernel_size = 4

mae_val_ep = np.zeros(len(LSTM_filters_set))
mae_train_ep = np.zeros(len(LSTM_filters_set))
i = 0
for LSTM_filters in LSTM_filters_set:
    model = tf.keras.models.Sequential([
          tf.keras.layers.Conv1D(filters = Conv_filters, kernel_size = Conv_kernel_size,
                      strides=1, padding="causal", activation="relu", input_shape=[None, n_outputs]),
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
          tf.keras.layers.Dense(n_outputs)
        ])
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    fitting = model.fit(features, epochs = n_epoch, verbose = 1, validation_data = val_features)
    
    mae_val_i = np.array(fitting.history['val_mae'])
    mae_val_ep[i] = np.min(mae_val_i[np.nonzero(mae_val_i)])
    min_position = fitting.history['val_mae'].index(mae_val_ep[i])
    mae_train_ep[i] = fitting.history['mae'][min_position]
    i += 1
plt.subplot(2,1,1)
plt.plot(LSTM_filters_set, mae_train_ep)
plt.subplot(2,1,2)
plt.plot(LSTM_filters_set, mae_val_ep)
LSTM_filters = 32

model = tf.keras.models.Sequential([
          tf.keras.layers.Conv1D(filters = Conv_filters, kernel_size = Conv_kernel_size,
                      strides=1, padding="causal", activation="relu", input_shape=[None, n_outputs]),
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_filters, return_sequences=True)),
          tf.keras.layers.Dense(n_outputs)
        ])
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
fitting = model.fit(features, epochs = 50, verbose = 1, validation_data = val_features)
mae = fitting.history['mae']
loss = fitting.history['loss']
epochs=range(len(loss))
plt.plot(epochs, mae, 'r')
plt.plot(epochs, fitting.history['val_mae'], 'r--')
plt.plot(epochs, loss, 'b')
plt.plot(epochs, fitting.history['val_loss'], 'b--')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["MAE", "val_MAE", "Loss", "val_Loss"])
last_month_forecast = model_forecast(model, valid_scaled, n_lag, batch_size)
last_month_forecast = forecast[-1,-1,:]
last_month_forecast = scaler.inverse_transform(np.expand_dims(last_month_forecast, axis = 0))
submission = pd.DataFrame({
        'ID': list_IDs,
        'item_cnt_month': np.squeeze(last_month_forecast)
    })
submission.head()
submission.loc[submission.item_cnt_month < 0, 'item_cnt_month'] = 0
submission = pd.merge(sales_test.ID, submission, on = ('ID'), how = 'left')
submission = submission.fillna(0)
submission.tail()
submission.to_csv('submission.csv', index=False)
