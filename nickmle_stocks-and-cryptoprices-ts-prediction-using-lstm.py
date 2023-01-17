# library imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import random
from sklearn.preprocessing import MinMaxScaler
from keras.models  import Sequential
from keras.layers import Dense, LSTM, Dropout
import shutil
def filesystem():
    # use this method, when run locally

    try:
        os.mkdir('data')
    except:
        pass
    try:
        os.mkdir("data/current")
    except:
        pass
    print("file system is all set")
#filesystem()
data = pd.read_csv('../input/random-stock-from-yahoo-finance/RIO.csv', index_col = 0, parse_dates = True)
training = data[['Adj Close']].values
# to rescale data for training
scaler =  MinMaxScaler(feature_range = (0, 1))
training_scaled = scaler.fit_transform(training)
training_data_points = 300
testing_scaled = training_scaled[-400-training_data_points:]
training_scaled = training_scaled[-2400:-400] 
print(len(training_scaled), len(testing_scaled))

def prepare_train_test(training_scaled, testing_scaled):
    X_train = []
    y_train = []
    for i in range(training_data_points, len(training_scaled)):
        X_train.append(training_scaled[i-training_data_points:i, 0])
        y_train.append(training_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = []
    y_test = []
    for i in range(training_data_points, len(testing_scaled)):
        X_test.append(testing_scaled[i-training_data_points:i, 0])
        y_test.append(testing_scaled[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test  = prepare_train_test(training_scaled, testing_scaled)
def get_model():
    model = Sequential()
    model.add(LSTM(units = 200, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(LSTM(units = 200, return_sequences = True))
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(LSTM(units = 100))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return model
model = get_model()
model.summary()
# training the model
history = model.fit(X_train, y_train, epochs= 100, batch_size = 60, validation_data=(X_test, y_test))
model.save_weights("lstm_stock_price_predict.h5")

history.history.keys()
def training_loss_graph(history):
    plt.plot(history.history['loss'], label = 'Training loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
training_loss_graph(history)

def get_predicted_INV_scaled(X_test):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    prices = scaler.inverse_transform([y_test])
    return predicted_prices, prices
predicted_prices, prices = get_predicted_INV_scaled(X_test)
def show_graph_result(prices, predicted_prices):
    # be mindful that the data is the global
    index = data.index.values[-len(prices[0]):]
    test_result = pd.DataFrame(columns = ["real", 'predicted'])
    test_result['real'] = prices[0]
    test_result['predicted'] = predicted_prices
    test_result.index = index
    test_result.plot(figsize= (16, 10))
    plt.title("Actual and Predicted prices from the test")
    plt.ylabel("Price")
    plt.xlabel("Time step")
    plt.show()
    
show_graph_result(prices, predicted_prices)
# future next --> 30 days what is going to happen
last_60_days = X_test[-1]
print(X_test.shape,last_60_days.shape)
#l.reshape(1, X_test[])
def get_number_days_predictions(days = 7):
    
    current = X_test[-1]
    print(current.shape)
    for i in range(days):
        predicted = model.predict(np.reshape(current, (1, current.shape[0], 1)))
        current1 = list(current[1:])
        current1.append(predicted[0])
        current = np.array(current1)
        print(" --- ")
    return current[-days:]


def generate_index(n):
    import datetime
    base = datetime.datetime.today()
    date_list  = [base + datetime.timedelta(days = x) for x in range(n)]
    return date_list


def get_result_frame(results, date_list):
    predict_df = pd.DataFrame(columns = ['predictions'])
    
    predict_df['predictions'] = results
    predict_df.index = date_list
    return predict_df


days_to_predict = 30

date_list = generate_index(days_to_predict)

results = get_number_days_predictions(days_to_predict)

results = scaler.inverse_transform(results).reshape(-1)   

predict_df  = get_result_frame(results, date_list)
predict_df.plot()
#1700*300*6
"""
High, Low, Close, Open, Volume, Adj Close


"""
"""
CNN  -- [nx6]--> image

"""