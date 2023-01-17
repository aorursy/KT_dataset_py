import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model

from keras.layers.core import Activation, Dropout, Dense, Flatten

from keras.layers.recurrent import GRU, LSTM

from keras.layers import Input

from keras.callbacks import EarlyStopping, ModelCheckpoint

import random as ran
# Load and prepare training data

data_file_path='../input/signal100hz/data_to_train_v4.csv'

home_data = pd.read_csv(data_file_path)



# Target variable

y = home_data['Frequency']

y = y[:10000]



# Features

syms = []

for i in range(0,100):

    syms.append(str(float(i)))

X = home_data[syms]

X = X[:10000]

# Data preparation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

X_train = np.reshape(np.array(X_train), (7500, 1, 100))

X_test = np.reshape(np.array(X_test), (2500, 1, 100))
model = Sequential()

model.add(LSTM(64,input_shape=(1,100),return_sequences=True))

model.add(Dropout(0.15))

model.add(LSTM(128,input_shape=(1,100),return_sequences=True))

model.add(Dropout(0.15))

model.add(LSTM(32,input_shape=(1,100),return_sequences=True))

model.add(Dropout(0.15))

model.add(LSTM(8, return_sequences=False))

model.add(Dropout(0.15))

model.add(Dense(1,activation="relu"))

model.compile(loss="mean_squared_error", optimizer="nadam")

print(model.summary())

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')

best_mod = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

print("The model has compiled.")

# Fitting the data

model.fit(X_train, y_train, batch_size=32, epochs=750, validation_data=(X_test, y_test),callbacks=[earlyStopping,best_mod])





# Evaluation

result_model = load_model('best_model.hdf5')

result = model.evaluate(X_test,y_test, verbose=1)

result_b = result_model.evaluate(X_test,y_test, verbose=1)

print("Mean Squared Error - early stopping: ",result)

print("Mean Squared Error - best model: ",result_b)
def generate_signal(samples, quantity):

    y_verif = []

    col_names = np.arange(0, (samples/2))

    col_names = np.append(col_names, 'Frequency')

    fsHz = samples  # Sampling frequency - 1s of signal -> samples = fsHz

    rng = np.arange(samples)

    data = pd.DataFrame(columns=col_names)

    for m in range(quantity):

            A = ran.randint(1, 20)

            fn = ran.randint(1,100)

            sig = A * np.sin(2 * np.pi * fn * (rng / samples)) + (A / 2) * np.sin(2 * np.pi * 50 * (rng / samples))

            for i, it in enumerate(sig):

                if it >= 0:

                    sig[i] = min(it, 0.9 * A)

                elif it < 0:

                    sig[i] = max(it, -0.9 * A)

            sig = sig + 0.1 * A + (0.1 * A * np.random.random(size=samples))

            transformed = np.fft.fft(sig, fsHz) / fsHz

            magTransformed = np.abs(transformed)

            finalFFT = np.fft.fftshift(magTransformed)

            finalFFT = finalFFT[int(fsHz / 2):int(fsHz)]

            finalFFT = np.append(finalFFT, fn)

            df = pd.DataFrame([np.transpose(finalFFT)], columns=col_names, index=None)

            data = data.append(df, ignore_index=True)

    return data
new_smps = generate_signal(200, 25)

new_smps.describe()
test_data = new_smps[syms]

test_values = new_smps['Frequency']

test_data = np.reshape(np.array(test_data), (25, 1, 100))

predicted_freq = model.predict(test_data)

pred_best = result_model.predict(test_data)

predicted_freq = np.reshape(np.array(predicted_freq), (25,1))

test_values = np.reshape(np.array(test_values), (25,1))

pred_best = np.reshape(np.array(pred_best), (25,1))



np.concatenate((predicted_freq, pred_best, test_values), axis=1)