# Library Imports

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import os

plt.style.use("ggplot")



from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout



dir=os.getcwd()



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Loading/Reading in the Data

df = pd.read_csv("/kaggle/input/BTC-USD.csv")



# Data Preprocessing

### Setting the datetime index as the date, only selecting the 'Close' column, then only the last 1000 closing prices.

df = df.set_index("Date")[['Close']].tail(1000)

df = df.set_index(pd.to_datetime(df.index))



# Normalizing/Scaling the Data (instead of BTC values like 9,321$, transform into 0<0.74<1)

scaler = MinMaxScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)





# Affiche 2 graphs, coût/epochs ET précision/epochs

def visualize_training_results(results):

    """

    Plots the loss and accuracy for the training and testing data

    """

    history = results.history

    plt.figure(figsize=(12,4))

    plt.plot(history['val_loss'])

    plt.plot(history['loss'])

    plt.legend(['val_loss', 'loss'])

    plt.title('Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.show()

    

    plt.figure(figsize=(12,4))

    plt.plot(history['val_accuracy'])

    plt.plot(history['accuracy'])

    plt.legend(['val_accuracy', 'accuracy'])

    plt.title('Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.show()



# renvoie pour un certain inter de tps (seq), 2 tableaux X et y

# exemple: X[0] correspond aux 30 1ers jours (si n_steps_in = 30)

#          y[0] correspond aux 30 jours d'après (si n_steps_out = 30)

def split_sequence(seq, n_steps_in, n_steps_out):

    """

    Splits the univariate time sequence

    """

    X, y = [], []

    

    for i in range(len(seq)):

        end = i + n_steps_in

        out_end = end + n_steps_out

        

        if out_end > len(seq):

            break

        

        seq_x, seq_y = seq[i:end], seq[end:out_end]

        

        X.append(seq_x)

        y.append(seq_y)

    

    return np.array(X), np.array(y)





def layer_maker(n_layers, n_nodes, activation, drop=None, d_rate=.5):

    """

    Create a specified number of hidden layers for an RNN

    Optional: Adds regularization option, dropout layer to prevent potential overfitting if necessary

    """

    

    # Creating the specified number of hidden layers with the specified number of nodes

    for x in range(1,n_layers+1):

        model.add(LSTM(n_nodes, activation=activation, return_sequences=True))



        # Adds a Dropout layer after every Nth hidden layer (the 'drop' variable)

        try:

            if x % drop == 0:

                model.add(Dropout(d_rate))

        except:

            pass

        

        

# How many periods looking back to train

n_per_in  = 30



# How many periods ahead to predict

n_per_out = 10



# Features (in this case it's 1 because there is only one feature: price)

n_features = 1



# Splitting the data into appropriate sequences

X, y = split_sequence(list(df.Close), n_per_in, n_per_out)



# Reshaping the X variable from 2D to 3D

X = X.reshape((X.shape[0], X.shape[1], n_features))





# Instantiating the model

model = Sequential()



# Activation

activ = "softsign"



# Input layer

model.add(LSTM(30, activation=activ, return_sequences=True, input_shape=(n_per_in, n_features)))



# Hidden layers

layer_maker(n_layers=6, n_nodes=12, activation=activ, drop=5)



# Final Hidden layer

model.add(LSTM(10, activation=activ))



# Output layer

model.add(Dense(n_per_out))



# Model summary

model.summary()



loss = 'mse'                  

optimizer="adam" 



model.compile(optimizer, loss)



res = model.fit(X, y, epochs=800, batch_size=32, validation_split=0.1)



model.save_weights("model_drop_5.h5")



visualize_training_results(res)