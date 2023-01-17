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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import time
# Set up random seed
seed = 17
np.random.seed(seed)
tf.random.set_seed(seed)
# Read raw csv data
data = pd.read_csv("../input/ctg-data/ctg_data_cleaned.csv")
data.head(5)
data['NSP'].value_counts()
data.describe()
NUM_CLASS = 3
def one_hot_encoding(dataframe):
    encoder = OneHotEncoder(sparse=False)
    dataframe = dataframe.values.reshape(-1, 1)
    data = encoder.fit_transform(dataframe)
    return encoder, data
def scale(dataframe):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(dataframe)
    return scale, data
X_dataframe = data.iloc[:, :21]
y_dataframe = data.iloc[:, -1]
scaler, X_dataframe = scale(X_dataframe)

# convert the labels from range [1,3] to [0,2]
# y_dataframe = y_dataframe - 1 
# y_dataframe = y_dataframe.values

encoder, y_dataframe = one_hot_encoding(y_dataframe)
data["NSP"].head(5)
y_dataframe[:5]
X_train, X_test, y_train, y_test = train_test_split(X_dataframe, y_dataframe, test_size=0.3, stratify=y_dataframe, random_state = 17)
def fit_baseline_model(X_train, y_train, X_test, y_test, batch_size, verbose, hidden_neurons, decay_rate, epochs):

    model = models.Sequential()

    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           input_shape=(X_train.shape[1],),
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    model.add(layers.Dense(NUM_CLASS, activation='softmax',
                          kernel_regularizer=keras.regularizers.l2(decay_rate)))

#     opt = keras.optimizers.SGD(learning_rate=0.01)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    
    model.compile(optimizer=opt,
                 loss= keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])

#     model.summary()

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=True,
                        validation_data=(X_test, y_test))
    
    return model, history
    
model, history = fit_baseline_model(X_train, y_train, X_test, y_test, 
                             batch_size=32, 
                             epochs=1000, 
                             verbose=0,
                             hidden_neurons=10,
                             decay_rate=1e-6)
plt.plot(history.history['accuracy'], label='Train_Accuracy')
plt.plot(history.history['val_accuracy'], label='Test_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Training Accuracy and Test Accuracy')
plt.legend()
plt.savefig('AQ1a')
plt.plot(history.history['loss'], label='Train_Error')
plt.plot(history.history['val_loss'], label='Test_Error')
plt.ylabel('Error')
plt.xlabel('No. of Epochs')
plt.title('Training Error and Test Error')
plt.legend()
plt.savefig('AQ1b')
batch_sizes = [4, 8, 16, 32, 64]
cvscore_per_bs = {}

for bs in batch_sizes:
    
    history_per_cv = []
    
    skf = StratifiedKFold(n_splits=5)
    
    y_train_categorical = encoder.inverse_transform(y_train).reshape(-1,)
    
    print(f"Performing 5 folds cross validation for batch size {bs}.")
    for train_index, test_index in skf.split(X_train, y_train_categorical):
        
        model, history = fit_baseline_model(X_train[train_index],
                                                 y_train[train_index],
                                                 X_train[test_index],
                                                 y_train[test_index],
                                                 batch_size=bs,
                                                 epochs=100,
                                                 decay_rate=1e-6,
                                                 hidden_neurons=10,
                                                verbose=2)
        history_per_cv.append(history.history)
    
    print(f"Done with 5 folds cross validation for batch size {bs}.")
    cvscore = np.zeros_like(history_per_cv[0]['accuracy'])
    
    for history in history_per_cv:
        cvscore = cvscore + history['val_accuracy']
    cvscore = cvscore / 5
    cvscore_per_bs[bs] = cvscore
for bs, cvscore in cvscore_per_bs.items():
    plt.plot(cvscore, label=f'batch_size_{bs}')

plt.ylabel('Cross Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Effect of Batch Size on CV Accuracy')
plt.legend(loc='lower right')
plt.savefig('AQ2a')
time_per_bs = {}

for bs in batch_sizes:
    
    time_begin = time.time()
    fit_baseline_model(X_train, y_train, X_test, y_test,
                       batch_size=bs,
                       epochs=100,
                       decay_rate=1e-6,
                       hidden_neurons=10,
                       verbose=2)
    time_end = time.time()
    
    time_per_bs[bs] = (time_end-time_begin)/100
x, y = zip(*time_per_bs.items())
plt.plot(x, y)
plt.ylabel('Average Time for One Epoch')
plt.xlabel('Batch Size')
plt.title('Effect of Batch Size on Training Time')
plt.savefig('AQ2b')
model, history = fit_baseline_model(X_train, y_train, X_test, y_test,
                       batch_size=32,
                       epochs=500,
                       decay_rate=1e-6,
                       hidden_neurons=10,
                       verbose=2)
plt.plot(history.history['accuracy'], label='Train_Accuracy')
plt.plot(history.history['val_accuracy'], label='Test_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epochs')
plt.ylim(bottom=0.75)
plt.title('Training Accuracy and Test Accuracy')
plt.legend(loc='lower right')
plt.savefig('AQ2c')
hidden_units = [5, 10, 15, 20, 25]

cvscore_per_hidden_units = {}

for hu in hidden_units:
    
    history_per_cv = []
    
    skf = StratifiedKFold(n_splits=5)
    
    y_train_categorical = encoder.inverse_transform(y_train).reshape(-1,)
    
    print(f"Performing 5 folds cross validation for hidden neurons {hu}.")
    for train_index, test_index in skf.split(X_train, y_train_categorical):
        
        model, history = fit_baseline_model(X_train[train_index],
                                                 y_train[train_index],
                                                 X_train[test_index],
                                                 y_train[test_index],
                                                 batch_size=32,
                                                 epochs=300,
                                                 decay_rate=1e-6,
                                                 hidden_neurons=hu,
                                                verbose=2)
        history_per_cv.append(history.history)
    print(f"Done with 5 folds cross validation for hidden neurons {hu}.")
    cvscore = np.zeros_like(history_per_cv[0]['accuracy'])
    
    for history in history_per_cv:
        cvscore = cvscore + history['val_accuracy']
    cvscore = cvscore / 5
    cvscore_per_hidden_units[hu] = cvscore
for hu, cvscore in cvscore_per_hidden_units.items():
    plt.plot(cvscore, label=f'hidden_neurons_{hu}')

plt.ylabel('Cross Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Effect of Hidden Neurons on CV Accuracy')
plt.legend(loc='lower right')
plt.savefig('AQ3a')
model, history = fit_baseline_model(X_train, y_train, X_test, y_test,
                       batch_size=32,
                       epochs=300,
                       decay_rate=1e-6,
                       hidden_neurons=25,
                       verbose=2)
plt.plot(history.history['accuracy'], label='Train_Accuracy')
plt.plot(history.history['val_accuracy'], label='Test_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epochs')
plt.ylim(bottom=0.75)
plt.title('Training Accuracy and Test Accuracy')
plt.legend(loc='lower right')
plt.savefig('AQ3c')
decay_params = [0, 1e-3, 1e-6, 1e-9, 1e-12]

cvscore_per_decay_param = {}

for decay_param in decay_params:
    
    history_per_cv = []
    
    skf = StratifiedKFold(n_splits=5)
    
    y_train_categorical = encoder.inverse_transform(y_train).reshape(-1,)
    
    print(f"Performing 5 folds cross validation for decay_rate {decay_param}.")
    for train_index, test_index in skf.split(X_train, y_train_categorical):
        
        model, history = fit_baseline_model(X_train[train_index],
                                                 y_train[train_index],
                                                 X_train[test_index],
                                                 y_train[test_index],
                                                 batch_size=32,
                                                 epochs=300,
                                                 decay_rate=decay_param,
                                                 hidden_neurons=25,
                                                verbose=0)
        history_per_cv.append(history.history)
    print(f"Done with 5 folds cross validation for decay_rate {decay_param}.")
    cvscore = np.zeros_like(history_per_cv[0]['accuracy'])
    
    for history in history_per_cv:
        cvscore = cvscore + history['val_accuracy']
    cvscore = cvscore / 5
    cvscore_per_decay_param[decay_param] = cvscore

for decay_param, cvscore in cvscore_per_decay_param.items():
    plt.plot(cvscore, label=f'decay_rate_{decay_param}')

plt.ylabel('Cross Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Effect of Decay Rate on CV Accuracy')
plt.legend(loc='lower right')
plt.savefig('AQ4a')
base_model, base_history = fit_baseline_model(X_train, y_train, X_test, y_test,
                       batch_size=32,
                       epochs=300,
                       decay_rate=1e-6,
                       hidden_neurons=25,
                       verbose=0)
plt.plot(history.history['accuracy'], label='Train_Accuracy')
plt.plot(history.history['val_accuracy'], label='Test_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epochs')
plt.ylim(bottom=0.75)
plt.title('Training Accuracy and Test Accuracy')
plt.legend(loc='lower right')
plt.savefig('AQ4c')
def fit_4_layer_model(X_train, y_train, X_test, y_test, batch_size, verbose, hidden_neurons, decay_rate, epochs):

    model = models.Sequential()

    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           input_shape=(X_train.shape[1],),
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           input_shape=(X_train.shape[1],),
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    model.add(layers.Dense(NUM_CLASS, activation='softmax',
                          kernel_regularizer=keras.regularizers.l2(decay_rate)))

#     opt = keras.optimizers.SGD(learning_rate=0.01)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    
    model.compile(optimizer=opt,
                 loss= keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])

#     model.summary()

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=True,
                        validation_data=(X_test, y_test))
    
    return model, history
four_layer_model, four_layer_history = fit_4_layer_model(X_train, y_train, X_test, y_test,
                       batch_size=32,
                       epochs=300,
                       decay_rate=1e-6,
                       hidden_neurons=10,
                       verbose=0)
plt.plot(four_layer_history.history['accuracy'], label='Train_Accuracy')
plt.plot(four_layer_history.history['val_accuracy'], label='Test_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epochs')
plt.ylim(bottom=0.75)
plt.title('Training Accuracy and Test Accuracy')
plt.legend(loc='lower right')
plt.savefig('AQ5a')
plt.plot(base_history.history['accuracy'], label='3_layer_Train_Accuracy')
plt.plot(base_history.history['val_accuracy'], label='3_layer_Test_Accuracy')
plt.plot(four_layer_history.history['accuracy'], label='4_layer_Train_Accuracy')
plt.plot(four_layer_history.history['val_accuracy'], label='4_layer_Test_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epochs')
plt.ylim(bottom=0.75)
plt.title('Training Accuracy and Test Accuracy')
plt.legend(loc='lower right')
plt.savefig('AQ5b')
base_history.history['loss']
plt.plot(base_history.history['loss'], label='3_layer_Train_Loss')
plt.plot(base_history.history['val_loss'], label='3_layer_Test_Loss')
plt.plot(four_layer_history.history['loss'], label='4_layer_Train_Loss')
plt.plot(four_layer_history.history['val_loss'], label='4_layer_Test_Loss')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epochs')
# plt.ylim(bottom=0.75)
plt.title('Training Loss and Test Loss')
plt.legend(loc='upper right')
plt.savefig('AQ5c')
