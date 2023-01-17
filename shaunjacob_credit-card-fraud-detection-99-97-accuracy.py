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


data_path = '../input/creditcardfraud/creditcard.csv'



credit_card_df = pd.read_csv(data_path)

credit_card_df = credit_card_df.sample(frac = 1).reset_index(drop = True) #shuffle the data by row

num_rows = len(credit_card_df)



#Split into training, validation, and test: 95, 2.5, 2.5 respectively

training_split = int(0.95 * num_rows)

validation_split = int((0.95 + 0.025) * num_rows)



#Now to separate the labels from the rest of the data: 1 for fraudulent transactions, 0 otherwise 

train_data_df = credit_card_df[:training_split]

y_train = train_data_df['Class']

x_train = train_data_df.drop('Class', axis = 1)

x_train = (x_train - x_train.mean()) / x_train.std() #Normalize train_data



validation_data_df = credit_card_df[training_split:validation_split]

y_val = validation_data_df['Class']

x_val = validation_data_df.drop('Class', axis = 1)

x_val = (x_val - x_val.mean()) / x_val.std() #Normalize validation_data



test_data_df = credit_card_df[validation_split:]

y_test = test_data_df['Class']

x_test = test_data_df.drop('Class', axis = 1)

x_test = (x_test - x_test.mean()) / x_test.std() #Normalize test data



#convert to numpy arrays

x_train = x_train.to_numpy()

y_train = y_train.to_numpy()



x_val = x_val.to_numpy()

y_val = y_val.to_numpy()



x_test = x_test.to_numpy()

y_test = y_test.to_numpy()



#One-hot encode labels

from tensorflow.keras.utils import to_categorical

to_categorical(y_train)

to_categorical(y_val)

to_categorical(y_test)
from tensorflow.keras import models, layers, optimizers, callbacks



callbacks_list = [

    callbacks.EarlyStopping(monitor = 'acc', patience = 1),

    callbacks.ModelCheckpoint(filepath = 'model_1.h5', monitor = 'val_loss', save_best_only = True),   

]



def build_model():

    model = models.Sequential()

    model.add(layers.Dense(128, activation = 'relu'))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation = 'relu'))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation = 'relu'))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(32, activation = 'relu'))

    model.add(layers.Dense(1, activation = 'sigmoid'))



    model.compile(optimizers.RMSprop(lr = 0.001),

                 loss = 'binary_crossentropy',

                 metrics = ['acc'])

    return model



model = build_model()

# train model

history = model.fit(x_train, y_train,

                   epochs = 20,

                    batch_size = 1024,

                   validation_data = (x_val, y_val),

                   callbacks = callbacks_list)
import matplotlib.pyplot as plt



#create history dictionary

history_dict = history.history



loss_values = history_dict['loss']

acc = history_dict['acc']

val_loss = history_dict['val_loss']

val_acc = history_dict['val_acc']



#plotting training and validation loss

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training Loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.figure()



#plotting training and validation accuracy

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')

plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.legend()

plt.show()
#Evaluate model on test data

model.evaluate(x_test, y_test)