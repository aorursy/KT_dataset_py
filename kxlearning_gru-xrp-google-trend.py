import csv

import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import GRU

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

from math import sqrt



from numpy.random import seed

from tensorflow import random



# Set the random seed

seed(1)

random.set_seed(2)



def iteration(batch_size: int, epoch: int, unit: int) -> float:

    raw_btc = pd.read_csv('https://raw.githubusercontent.com/KunXie/CIS-Paper-dataset/master/XRP-USD.csv')

    raw_gt = pd.read_csv('https://raw.githubusercontent.com/KunXie/CIS-Paper-dataset/master/XRP-GOOGLE.csv')



    btc = raw_btc.copy()

    btc['Date'] = pd.to_datetime(btc['Date'])

    btc.index = btc.Date    # set the index

    # btc = btc.loc['2015-01-01':'2019-12-31']



    gt = raw_gt.copy()

    gt['Date'] = pd.to_datetime(gt['Date'])

    gt.index = gt.Date

    # gt = gt.loc['2015-01-01':'2019-12-31']



    # shape: (1826, 2)

    df = pd.merge(btc, gt, left_index=True, right_index=True)[['Close', 'XRP']].values



    prediction_days = 400

    # (1426, 2) and (400, 2)

    train, test = df[:-prediction_days], df[-prediction_days:]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train)



    # fixed parameters

    activation_function = 'sigmoid'

    optimizer = 'adam'

    loss_function = 'mean_squared_error'



    model = Sequential()

    model.add(GRU(units=unit, activation=activation_function, input_shape=(1, 2)))

    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss=loss_function)



    x, y = train_scaled[:-1], train_scaled[1:, 0]

    x = x.reshape(len(x), 1, 2)

    y = y.reshape(len(y), 1)

    model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose=0)



    # difference here

    new_test = np.concatenate((train[-1:], test))

    new_test_scaled = scaler.transform(new_test)

    x_, y_ = new_test_scaled[:-1], test[:, 0]

    x_ = x_.reshape(len(x_), 1, 2)

    y_ = y_.reshape(len(y_), 1)

    preds_scaled = model.predict(x_) # (400, 1)



    # more modifications for inverse transform

    temp = np.concatenate((preds_scaled, new_test_scaled[1:, 1].reshape(len(test), 1)), axis=1)

    preds = scaler.inverse_transform(temp)



    rmse = sqrt(mean_squared_error(y_[:, 0], preds[:, 0]))

    return rmse





batch_sizes = [1, 2, 3, 4, ]

epochs = [10, 20, 50, 100, ]

unit_sizes = [1, 2, 3, 4, ]



# Create a csv file

filename = 'GRU-XRP-GOOGLE-TREND.csv'

with open(filename, 'w') as csv_file:

    file_writer = csv.writer(csv_file, delimiter=',')

    file_writer.writerow(['rmse', 'batch_size', 'epoch', 'unit'])  # set the heading



for batch_size in batch_sizes:

    for epoch in epochs:

        for unit in unit_sizes:

            rmse = iteration(batch_size, epoch, unit)



            # write into csv file

            with open(filename, 'a') as csv_file:

                file_writer = csv.writer(csv_file, delimiter=',')

                file_writer.writerow([rmse, batch_size, epoch, unit])  # set the title

            print(str(rmse) + ", " + str(batch_size) + ", " + str(epoch) + ", " + str(unit))