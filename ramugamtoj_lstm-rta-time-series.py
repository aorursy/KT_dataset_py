import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../input/epolicija-eismo-ivykiai/sortedAccidents.csv')
data.head() 
# Let's load the required libs.
# We'll be using the Tensorflow backend (default).
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
# Get the raw data values from the pandas data frame.
data_raw = data[["1"]].astype("float32")

# We apply the MinMax scaler from sklearn
# to normalize data in the (0, 1) interval.
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)

# Print a few values.
dataset[0:5]
TRAIN_SIZE = 0.67

train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Number of entries (training set, test set): " + str((len(train), len(test))))

# FIXME: This helper function should be rewritten using numpy's shift function. See below.
def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))
# Create test and training sets for one-step-ahead regression.
window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)

# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)
def compile_model(train_X, train_Y, window_size = 1):
    model = Sequential()
    
    model.add(LSTM(4, 
                   input_shape = (1, window_size)))
    model.add(Dense(1))
    model.compile(loss = "huber_loss", optimizer = "Adam")
    return(model)
model1 = compile_model(train_X, train_Y, window_size)
model1.summary()

#import keras.backend as K
#print('LR', K.eval(model1.optimizer.lr))
# Fit the first model.
model1.fit(train_X, 
          train_Y, 
              epochs = 100, 
              batch_size = 1, 
              verbose = 2)
# Save model
model1.save_weights('./model')


#model1.load_weights('../input/models/model-msle-100e-adam')
def predict_and_score_mse(model, X, Y):
    # Make predictions on the original scale of the data.
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    score = mean_squared_error(orig_data[0], pred[:, 0])
    return(score, pred)

def predict_and_score_rmse(model, X, Y):
    mse_score, pred = predict_and_score_mse(model, X, Y)
    rmse_score = math.sqrt(mse_score)
    return(rmse_score, pred)

rmse_train, train_predict = predict_and_score_rmse(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score_rmse(model1, test_X, test_Y)

mse_train, train_predict = predict_and_score_mse(model1, train_X, train_Y)
mse_test, test_predict = predict_and_score_mse(model1, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)

print("Training data score: %.2f MSE" % mse_train)
print("Test data score: %.2f MSE" % mse_test)
# Start with training predictions.
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# Add test predictions.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict

# Create the plot.
plt.figure(figsize = (15, 5))
plt.xticks(range(0, len(train_predict) + len(test_predict), 365))
plt.plot(scaler.inverse_transform(dataset), label = "Tikri duomenys")
plt.plot(train_predict_plot, label = "Mokymosi duomenų prognozė")
plt.plot(test_predict_plot, label = "Validacijos duomenų prognozė")
plt.xlabel("Dienos (nuo 2013-01-01)")
plt.ylabel("Eismo įvykiai")
plt.title("Tikrų duomenų su modelio prognozėmis palyginimas")
plt.legend()
plt.show()
