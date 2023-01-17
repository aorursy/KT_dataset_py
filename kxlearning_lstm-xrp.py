import pandas as pd



import csv

from math import sqrt

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler



from numpy.random import seed

from tensorflow import random



# Set the random seed

seed(1)

random.set_seed(2)





def iteration(batch_size: int, epoch: int, unit: int) -> float:

    # make the data start from 2015-01-01

    df = pd.read_csv('https://raw.githubusercontent.com/KunXie/CIS-Paper-dataset/master/XRP-USD.csv')

    df = df['Close']



    prediction_days = 400

    train, test = df[:-prediction_days], df[-prediction_days:]

    scaler = MinMaxScaler()

    train_arr = train.values.reshape(len(train), 1)

    train_scaled = scaler.fit_transform(train_arr)



    # fixed variables

    activation_function = 'sigmoid'

    optimizer = 'adam'

    loss_function = 'mean_squared_error'



    model = Sequential()

    # Add LSTM layer

    model.add(LSTM(units=unit, activation=activation_function, input_shape=(None, 1)))

    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss=loss_function)



    x, y = train_scaled[:-1], train_scaled[1:]

    x = x.reshape(len(x), 1, 1)

    y = y.reshape(len(y), 1)

    model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose=0)



    # difference here

    new_test = train[-1:].append(test).values.reshape(len(test) + 1, 1)

    new_test_scaled = scaler.transform(new_test)

    x_, y_ = new_test_scaled[:-1], test.values  # y_ is non-scaled test

    x_ = x_.reshape(len(x_), 1, 1)

    y_ = y_.reshape(len(y_), 1)



    preds_scaled = model.predict(x_)

    preds = scaler.inverse_transform(preds_scaled)



    rmse = sqrt(mean_squared_error(y_[:, 0], preds[:, 0]))

    return rmse





batch_sizes = [1, 2, 3, 4, ]

epochs = [10, 20, 50, 100, ]

unit_sizes = [1, 2, 3, 4, ]



# Create a csv file

filename = 'LSTM-XRP-result.csv'

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
