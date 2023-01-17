# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import matplotlib.pyplot as plt

from math import sqrt, ceil

from timeit import default_timer as timer



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint



import os

import tensorflow as tf
# Change these

choice = 0        



data_filename = f"/kaggle/input/traffic-signs-preprocessed/data{choice}.pickle"

pkl_filename = f"/saved_models/data{choice}_model.pkl"



plot_acc = f"/training_plots/data{choice}_acc.png"

plot_loss = f"/training_plots/data{choice}_loss.png"



file_training_validation_results = f"/results/train_validate/data{choice}_train_validate_results.txt"

file_testing_results = f"/results/test/data{choice}_test_results.txt"

best_model_results = f"/results/best_models/data{choice}_best_model.txt"

# Change these: ends
epochs = 15

epoch_step_size = 3



activations = ["sigmoid","tanh","relu"]

dropouts = [0.1,0.3,0.5]

optimizers = ['adam','sgd']

neurons = [32, 64, 128]
annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)

mc = tf.keras.callbacks.ModelCheckpoint(f"model_data{choice}.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)



models = {}

history = {}



best_activation = -1

best_dropout = -1

best_optimizer = ""

best_neuron = -1

best_acc = 0
def plot_model(history, activation, dropout, optimizer, neurons):

    plt.figure(1)



    plt.plot(history.history['acc'], 'b', label='Training Accuracy')

    plt.plot(history.history['val_acc'], 'r', label='Validation Accuracy')

    plt.legend(loc='upper right')

    plt.ylabel('Accuracy')

    plt.xlabel('Epochs')

    plt.xticks(np.arange(0,epochs,step=epoch_step_size))

    plt.title('Accuracy Curves for Activation: {0}, Dropout: {1}, Optimizer: {2}, Neurons: {3}'.format(activation,dropout,optimizer,neuron))

    plt.savefig(plot_acc, bbox_inches='tight')



    plt.figure(2)

    plt.plot(history.history['loss'], 'b', label='Training Loss')

    plt.plot(history.history['val_loss'], 'r', label='Validation Loss')

    plt.legend(loc='upper right')

    plt.ylabel('Loss')

    plt.xlabel('Epochs')

    plt.xticks(np.arange(0,epochs,step=epoch_step_size))

    plt.title('Loss Curves for Activation: {0}, Dropout: {1}, Optimizer: {2}, Neurons: {3}'.format(activation,dropout,optimizer,neuron))

    plt.savefig(plot_loss, bbox_inches='tight')
def create_model(activation="relu", dropout=0.0,optimizer="adam",neurons=128, channel=1):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(neurons, kernel_size=3, padding='same', activation=activation, input_shape=(32, 32, channel)))

    model.add(tf.keras.layers.MaxPool2D(pool_size=2))

    model.add(tf.keras.layers.Dropout(dropout))



    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation=activation))

    model.add(tf.keras.layers.Dense(256, activation=activation))

    model.add(tf.keras.layers.Dense(43, activation='softmax'))

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model
# Execution starts here



# with tf.device('/device:GPU:0'):

with open(data_filename, 'rb') as f:

    data = pickle.load(f, encoding='latin1')  # dictionary type



y_train = data['y_train']

y_val = data['y_validation']

y_test = data['y_test']



# Making channels come at the end

X_train = data['x_train'].transpose(0, 2, 3, 1)

X_val = data['x_validation'].transpose(0, 2, 3, 1)

X_test = data['x_test'].transpose(0, 2, 3, 1)





CHANNEL = X_train.shape[-1]       # 1 for grayscale, 3 for RGB





# data["x_train"] = np.concatenate((data["x_train"],data["x_validation"]),axis=0)

# data["y_train"] = np.concatenate((data["y_train"],data["y_validation"]),axis=0)



training_file = open(file_training_validation_results, "w+")



for activation in activations:

    models[activation] = {}

    for dropout in dropouts:

        models[activation][dropout] = {}

        for optimizer in optimizers:

            models[activation][dropout][optimizer] = {}

            for neuron in neurons:

                train_msg = "Training with params {0}, {1}, {2}, {3}".format(activation,dropout,optimizer,neuron)

                print(train_msg)

                model = create_model(activation=activation,dropout=dropout,optimizer=optimizer,neurons=neuron, channel=CHANNEL)

                model.fit(data['x_train'], data['y_train'],batch_size=512, epochs = epochs, validation_split=0.3,callbacks=[annealer, es, mc])

                models[activation][dropout][optimizer][neuron] = model

                train_result = "Training Accuracy = {0}, Validation Accuracy = {1}".format(model.history.history["acc"][-1],model.history.history["val_acc"][-1])

                training_file.write(train_msg + "\n" + train_result + "\n========\n")



training_file.close()



testing_file = open(file_testing_results, "w+")



"""# Calculating accuracy with testing dataset"""





for activation in activations:

    for dropout in dropouts:

        for optimizer in optimizers:

            for neuron in neurons:

                temp = models[activation][dropout][optimizer][neuron].predict(data['x_test'])

                temp = np.argmax(temp, axis=1)



                temp = np.mean(temp == data['y_test'])

                if temp > best_acc:

                    best_acc = temp

                    best_activation = activation

                    best_dropout = dropout

                    best_optimizer = optimizer

                    best_neuron = neuron



                test_result = "Test Accuracy = {0} for the model: Activation={1}, Dropout={2}, Optimizer={3}, Neurons={4}".format(temp,activation,dropout,optimizer,neuron)

testing_file.write(test_result + "\n========\n")

testing_file.close()



best_m = open(best_model_results, "w+")

best_result = "BEST MODEL\nTest Accuracy = {0} for the model: Activation={1}, Dropout={2}, Optimizer={3}, Neurons={4}".format(best_acc,best_activation,best_dropout,best_optimizer,best_neuron)

best_m.write(best_result)

best_m.close()



best_model = models[best_activation][best_dropout][best_optimizer][best_neuron]

plot_model(best_model.history,best_activation, best_dropout, best_optimizer, best_neuron)



with open(pkl_filename, 'wb') as file:

  pickle.dump(best_model, file)
