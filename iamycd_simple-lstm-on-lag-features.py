import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import pickle



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)



seed = 13
data = pd.read_pickle('../input/eda-and-feature-engineering-data-exported/lagonly.pkl')

data.dtypes
X_train = data[(data.date_block_num < 32) & (data.date_block_num > 12)].drop(['item_cnt_month'], axis=1)

Y_train = data[(data.date_block_num < 32) & (data.date_block_num > 12)]['item_cnt_month']

X_val = data[(data.date_block_num > 31) & (data.date_block_num < 34)].drop(['item_cnt_month'], axis=1)

Y_val = data[(data.date_block_num > 31) & (data.date_block_num < 34)]['item_cnt_month']



X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
# reshape into [samples, timesteps, features]

def reshape_lag_dateset(dataset):

    altered_dataset = []

    for row in dataset.itertuples():

        altered_dataset.append([

            [row.item_cnt_month_lag_1,row.item_cnt_month_lag_2,

             row.item_cnt_month_lag_3,row.item_cnt_month_lag_6,row.item_cnt_month_lag_12],

            [row.average_daily_lag_1,row.average_daily_lag_2,

             row.average_daily_lag_3,row.average_daily_lag_6,row.average_daily_lag_12],

            [row.no_of_t_lag_1, row.no_of_t_lag_2,

             row.no_of_t_lag_3, row.no_of_t_lag_6,row.no_of_t_lag_12 ],

            [row.item_category_cnt_lag_1, row.item_category_cnt_lag_2,

             row.item_category_cnt_lag_3, row.item_category_cnt_lag_6,row.item_category_cnt_lag_12],

            [row.avg_category_price_lag_1, row.avg_category_price_lag_2,

             row.avg_category_price_lag_3, row.avg_category_price_lag_6,row.avg_category_price_lag_12],

            [row.category_price_change_count_lag_1, row.category_price_change_count_lag_2,

             row.category_price_change_count_lag_3, row.category_price_change_count_lag_6,row.category_price_change_count_lag_12],

            [row.avg_category_item_cnt_month_lag_1, row.avg_category_item_cnt_month_lag_2,

             row.avg_category_item_cnt_month_lag_3, row.avg_category_item_cnt_month_lag_6,row.avg_category_item_cnt_month_lag_12 ],

        ])

    return np.array(altered_dataset)

    

X_train_values = np.reshape(reshape_lag_dateset(X_train), (X_train.shape[0], 5, 7))

X_val_values = np.reshape(reshape_lag_dateset(X_val), (X_val.shape[0], 5, 7))

X_test_values = np.reshape(reshape_lag_dateset(X_test), (X_test.shape[0], 5, 7))



batch_size = 10000

model = Sequential()

model.add(LSTM(12, batch_input_shape=(None, 5, 7), return_sequences=True))

model.add(LSTM(4, batch_input_shape=(None, 5, 7)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train_values, Y_train, epochs=100, batch_size=batch_size, verbose=2)



val_predictions = model.predict(X_val_values)

error = np.sqrt(mean_squared_error(Y_val, val_predictions))



print('RMSE:',error)
predictions = model.predict(X_test_values)



submission = pd.DataFrame({

    "ID": X_test.index, 

    "item_cnt_month": predictions.reshape(-1)

})

submission.to_csv('predictions.csv', index=False)