# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import packages 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import time
# Read csv data from file
data = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
data.head()
len(data)
# Set up random seed
seed = 17
np.random.seed(seed)
tf.random.set_seed(seed)
# Normalise the input features
def scale(dataframe):
    scaler = StandardScaler(with_mean=True, with_std=True)
    data = scaler.fit_transform(dataframe)
    return scaler, data
# Prepare for raw X and y data.
X_dataframe = data.iloc[:, 1:-1]
y_dataframe = data.iloc[:, -1]
scaler, X_dataframe = scale(X_dataframe)
y_dataframe = y_dataframe.values
# Split the data to 70:30 for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_dataframe, y_dataframe, test_size=0.3, random_state = 17)
def fit_baseline_model(X_train, y_train, X_test, y_test, batch_size, verbose, hidden_neurons, decay_rate, epochs):
    
    print("Constructing the model...")

    model = models.Sequential()
    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           input_shape=(X_train.shape[1],),
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    model.add(layers.Dense(1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    opt = keras.optimizers.Adam(learning_rate=1e-3)
#     opt = keras.optimizers.SGD(learning_rate=1e-3)
    
    print("Compile the model...")
    model.compile(optimizer=opt,
                 loss= keras.losses.MeanSquaredError(),
                 metrics=[keras.metrics.RootMeanSquaredError()])

    model.summary()

    print("Training the model...")
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(X_test, y_test))
    print("Done training the model.")
    return model, history
     
base_model, base_history = fit_baseline_model(X_train, y_train, X_test, y_test,
                                              batch_size=8,
                                              epochs=1000,
                                              verbose=0,
                                              hidden_neurons=10,
                                              decay_rate=1e-3)
plt.plot(base_history.history['root_mean_squared_error'], label='Train_RMSE')
plt.plot(base_history.history['val_root_mean_squared_error'], label='Test_RMSE')
plt.ylabel('Root Mean Squared Error')
plt.ylim(top=0.2)
plt.xlabel('No. of Epochs')
plt.title('Training RSME and Test RMSE')
plt.legend()
plt.savefig('BQ1a1')
plt.plot(base_history.history['loss'], label='Train_Error')
plt.plot(base_history.history['val_loss'], label='Test_Error')
plt.ylabel('Mean Squared Error')
plt.ylim(top=0.02, bottom=0)
plt.xlabel('No. of Epochs')
plt.title('Training MSE and Test MSE')
plt.legend()
plt.savefig('BQ1a2')
# Plot predicted values and target values for 50 test samples
predicted_50 = base_model.predict(X_test[:50])
plt.plot(predicted_50,'o-', label='Predicted Value')
plt.plot(y_test[:50],'o-', label='Groud Truth')
plt.ylabel('Admission Prob.')
# plt.ylim(top=0.02)
# plt.xlabel('No. of Epochs')
# plt.title('Training Error and Test Error')
plt.legend()
plt.savefig('BQ1c')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
# Perform a basic linear regression on the original data
reg = LinearRegression()
# reg.fit(X_dataframe, y_dataframe)
# reg.score(X_dataframe, y_dataframe)
six_selector = RFE(reg, n_features_to_select=6, step=1)
six_selector.fit(X_dataframe, y_dataframe)
print(six_selector.support_)
print(six_selector.ranking_)
five_selector = RFE(reg, n_features_to_select=5, step=1)
five_selector.fit(X_dataframe, y_dataframe)
print(five_selector.support_)
print(five_selector.ranking_)
# Training the model with 6 input features
model_six, history_six = fit_baseline_model(X_train[:, six_selector.support_], y_train, X_test[:, six_selector.support_], y_test,
                  batch_size=8,
                  epochs=1000,
                  verbose=0,
                  hidden_neurons=10,
                  decay_rate=1e-3)
# Training the model for 5 input features
model_five, history_five = fit_baseline_model(X_train[:, five_selector.support_], y_train, X_test[:, five_selector.support_], y_test,
                  batch_size=8,
                  epochs=1000,
                  verbose=0,
                  hidden_neurons=10,
                  decay_rate=1e-3)
plt.plot(base_history.history['root_mean_squared_error'], label='7_input_features')
# plt.plot(history.history['val_root_mean_squared_error'], label='7_Test_RMSE')
plt.plot(history_six.history['root_mean_squared_error'], label='6_input_features')
# plt.plot(history_six.history['val_root_mean_squared_error'], label='6_Test_RMSE')
plt.plot(history_five.history['root_mean_squared_error'], label='5_input_features')
# plt.plot(history_five.history['val_root_mean_squared_error'], label='5_Test_RMSE')

plt.ylabel('RMSE')
plt.ylim(top=0.1)
plt.xlim(right=1000)
plt.xlabel('No. of Epochs')
plt.title('Training Root Mean Squared Error')
plt.legend()
plt.savefig('BQ2-1')
# plt.plot(base_history.history['root_mean_squared_error'], label='7_Train_RMSE')
plt.plot(base_history.history['val_root_mean_squared_error'], label='7_input_features')
# plt.plot(history_six.history['root_mean_squared_error'], label='6_Train_RMSE')
plt.plot(history_six.history['val_root_mean_squared_error'], label='6_input_features')
# plt.plot(history_five.history['root_mean_squared_error'], label='5_Train_RMSE')
plt.plot(history_five.history['val_root_mean_squared_error'], label='5_input_features')

plt.ylabel('RMSE')
plt.ylim(top=0.1)
plt.xlim(right=1000)
plt.xlabel('No. of Epochs')
plt.title('Test Root Mean Squared Error ')
plt.legend()
plt.savefig('BQ2-2')
print(f"Minimum MSE for a model of 7 features is: {min(base_history.history['root_mean_squared_error'])}")
print(f"Minimum MSE for a model of 6 features is: {min(history_six.history['root_mean_squared_error'])}")
print(f"Minimum MSE for a model of 5 features is: {min(history_five.history['root_mean_squared_error'])}")
print(f"Minimum RMSE for a model of 7 features is: {min(base_history.history['val_root_mean_squared_error'])}")
print(f"Minimum RMSE for a model of 6 features is: {min(history_six.history['val_root_mean_squared_error'])}")
print(f"Minimum RMSE for a model of 5 features is: {min(history_five.history['val_root_mean_squared_error'])}")
def fit_four_layer_model(X_train, y_train, X_test, y_test, batch_size, verbose, hidden_neurons, decay_rate, epochs, dropout_rate):
    
    print("Constructing the model...")

    model = models.Sequential()
    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           input_shape=(X_train.shape[1],),
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    if(dropout_rate > 0):
        model.add(layers.Dropout(rate=dropout_rate, seed=seed))
    
    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    if(dropout_rate > 0):
        model.add(layers.Dropout(rate=dropout_rate, seed=seed))
    
    model.add(layers.Dense(1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    opt = keras.optimizers.Adam(learning_rate=1e-3)
#     opt = keras.optimizers.SGD(learning_rate=1e-3)
    
    print("Compile the model...")
    model.compile(optimizer=opt,
                 loss= keras.losses.MeanSquaredError(),
                 metrics=[keras.metrics.RootMeanSquaredError()])

    model.summary()

    print("Training the model...")
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(X_test, y_test))
    print("Done training the model.")
    return model, history
     
model_4layer, model_4layer_history = fit_four_layer_model(X_train, y_train, 
                                                               X_test, y_test,
                                                               batch_size=8,
                                                               epochs=1000,
                                                               verbose=0,
                                                               hidden_neurons=50,
                                                               dropout_rate = -1,
                                                               decay_rate=0)
plt.plot(model_4layer_history.history['root_mean_squared_error'], label='Train_RMSE')
plt.plot(model_4layer_history.history['val_root_mean_squared_error'], label='Test_RMSE')
plt.ylabel('Root Mean Squared Error')
plt.ylim(top=0.2)
plt.xlabel('No. of Epochs')
plt.title('4 Layer Model')
plt.legend()
plt.savefig('BQ3-1')
model_4layer_dropout, model_4layer_dropout_history = fit_four_layer_model(X_train, y_train, 
                                                               X_test, y_test,
                                                               batch_size=8,
                                                               epochs=1000,
                                                               verbose=0,
                                                               hidden_neurons=50,
                                                               dropout_rate = 0.2,
                                                               decay_rate=0)
plt.plot(model_4layer_dropout_history.history['root_mean_squared_error'], label='Train_RMSE')
plt.plot(model_4layer_dropout_history.history['val_root_mean_squared_error'], label='Test_RMSE')
plt.ylabel('Root Mean Squared Error')
plt.ylim(top=0.2)
plt.xlabel('No. of Epochs')
plt.title('4 Layer Model with Dropout')
plt.legend()
plt.savefig('BQ3-2')
def fit_five_layer_model(X_train, y_train, X_test, y_test, batch_size, verbose, hidden_neurons, decay_rate, epochs, dropout_rate):
    
    print("Constructing the model...")

    model = models.Sequential()
    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           input_shape=(X_train.shape[1],),
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    if(dropout_rate > 0):
        model.add(layers.Dropout(rate=dropout_rate, seed=seed))
    
    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    if(dropout_rate > 0):
        model.add(layers.Dropout(rate=dropout_rate, seed=seed))
        
    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    if(dropout_rate > 0):
        model.add(layers.Dropout(rate=dropout_rate, seed=seed))
    
    model.add(layers.Dense(1, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    opt = keras.optimizers.Adam(learning_rate=1e-3)
#     opt = keras.optimizers.SGD(learning_rate=1e-3)
    
    print("Compile the model...")
    model.compile(optimizer=opt,
                 loss= keras.losses.MeanSquaredError(),
                 metrics=[keras.metrics.RootMeanSquaredError()])

    model.summary()

    print("Training the model...")
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(X_test, y_test))
    print("Done training the model.")
    return model, history
     
model_5layer, model_5layer_history = fit_five_layer_model(X_train, y_train, 
                                                               X_test, y_test,
                                                               batch_size=8,
                                                               epochs=1000,
                                                               verbose=0,
                                                               hidden_neurons=50,
                                                               dropout_rate = -1,
                                                               decay_rate=0)
plt.plot(model_4layer_dropout_history.history['root_mean_squared_error'], label='Train_RMSE')
plt.plot(model_4layer_dropout_history.history['val_root_mean_squared_error'], label='Test_RMSE')
plt.ylabel('Root Mean Squared Error')
plt.ylim(top=0.2)
plt.xlabel('No. of Epochs')
plt.title('5 Layer Model')
plt.legend()
plt.savefig('BQ3-3')
model_5layer_dropout, model_5layer_dropout_history = fit_five_layer_model(X_train, y_train, 
                                                               X_test, y_test,
                                                               batch_size=8,
                                                               epochs=1000,
                                                               verbose=0,
                                                               hidden_neurons=50,
                                                               dropout_rate = 0.2,
                                                               decay_rate=0)
plt.plot(model_4layer_dropout_history.history['root_mean_squared_error'], label='Train_RMSE')
plt.plot(model_4layer_dropout_history.history['val_root_mean_squared_error'], label='Test_RMSE')
plt.ylabel('Root Mean Squared Error')
plt.ylim(top=0.2)
plt.xlabel('No. of Epochs')
plt.title('5 Layer Model with Dropout')
plt.legend()
plt.savefig('BQ3-4')
plt.plot(base_history.history['val_root_mean_squared_error'], label='3 layer model')
plt.plot(model_4layer_history.history['val_root_mean_squared_error'], label='4 layer model')
plt.plot(model_4layer_dropout_history.history['val_root_mean_squared_error'], label='4 layer + dropout')
plt.plot(model_5layer_history.history['val_root_mean_squared_error'], label='5 layer model')
plt.plot(model_5layer_dropout_history.history['val_root_mean_squared_error'], label='5 layer + dropout')
plt.ylabel('Root Mean Squared Error')
plt.ylim(top=0.2)
plt.xlabel('No. of Epochs')
plt.title('Test Error for all models')
plt.legend()
plt.savefig('BQ3-5')
plt.plot(base_history.history['root_mean_squared_error'], label='3 layer model')
plt.plot(model_4layer_history.history['root_mean_squared_error'], label='4 layer model')
plt.plot(model_4layer_dropout_history.history['root_mean_squared_error'], label='4 layer + dropout')
plt.plot(model_5layer_history.history['root_mean_squared_error'], label='5 layer model')
plt.plot(model_5layer_dropout_history.history['root_mean_squared_error'], label='5 layer + dropout')
plt.ylabel('Root Mean Squared Error')
plt.ylim(top=0.2)
plt.xlabel('No. of Epochs')
plt.title('Train Error for all models')
plt.legend()
plt.savefig('BQ3-6')
model_4layer_l2, model_4layer_l2_history = fit_four_layer_model(X_train, y_train, 
                                                               X_test, y_test,
                                                               batch_size=8,
                                                               epochs=500,
                                                               verbose=0,
                                                               hidden_neurons=50,
                                                               dropout_rate = 0.2,
                                                               decay_rate=1e-9)
plt.plot(base_history.history['val_root_mean_squared_error'], label='3 layer model test')
plt.plot(model_4layer_l2_history.history['val_root_mean_squared_error'], label='4 layer model test')
plt.plot(base_history.history['root_mean_squared_error'], label='3 layer model train')
plt.plot(model_4layer_l2_history.history['root_mean_squared_error'], label='4 layer model train')
plt.ylabel('Root Mean Squared Error')
plt.ylim(top=0.1)
plt.xlim(right=500)
plt.xlabel('No. of Epochs')
plt.title('Test Error')
plt.legend()
plt.savefig('BQ3-7')
print(f"Minimum RMSE for a 3 layer model is: {min(base_history.history['val_root_mean_squared_error'])}")
print(f"Minimum RMSE for a 4 layer model is: {min(model_4layer_history.history['val_root_mean_squared_error'])}")
print(f"Minimum RMSE for a 4 layer model with dropout is: {min(model_4layer_dropout_history.history['val_root_mean_squared_error'])}")
print(f"Minimum RMSE for a 5 layer model is: {min(model_5layer_history.history['val_root_mean_squared_error'])}")
print(f"Minimum RMSE for a 5 layer model with dropout is: {min(model_5layer_dropout_history.history['val_root_mean_squared_error'])}")

