!git clone https://github.com/dozeeio/snoring
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras import layers, models



import librosa

import librosa.display



from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier 
telemetry = pd.read_csv("/kaggle/working/snoring/test_data/test2.csv", header=None, squeeze=False)[:20_000]

print(telemetry.shape)

telemetry.plot()



telemetry2 = pd.read_csv("/kaggle/working/snoring/test_data/test2.csv", header=None, squeeze=False)

# telemetry2.plot()



def calc_stft_feats(telemetry):

    S = librosa.core.stft(telemetry.values.reshape(-1).astype(np.float), 

                      n_fft=128, 

                      hop_length=1,

                      win_length=100)

    spect_db = librosa.amplitude_to_db(S, ref=np.max)

    return spect_db



def plot_stft(telemetry):

    

    spect_db = calc_stft_feats(telemetry)



    fig, ax = plt.subplots(figsize=(24, 12))

    img = librosa.display.specshow(spect_db, y_axis='log', ax=ax)

    ax.set_title('Power spectrogram')

    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    

plot_stft(telemetry)
spect_db = calc_stft_feats(telemetry)

spect_db.shape
feats_touse_index = spect_db.var(1) > 10
TIME_STEPS = 32



# Generated training sequences for use in the model.

def create_sequences(values, time_steps=TIME_STEPS):

    output = []

    for i in range(len(values) - time_steps):

        output.append(values[i : (i + time_steps)])

    return np.stack(output)





x_train = create_sequences(spect_db.transpose()[:,feats_touse_index])

print("Training input shape: ", x_train.shape)
def get_AE_model(x_train):

    model = keras.Sequential(

        [

            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),

            layers.Conv1D(

                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"

            ),

            layers.Dropout(rate=0.2),

            layers.Conv1D(

                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"

            ),

            layers.Conv1DTranspose(

                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"

            ),

            layers.Dropout(rate=0.2),

            layers.Conv1DTranspose(

                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"

            ),

            layers.Conv1DTranspose(filters=x_train.shape[2], kernel_size=7, padding="same"),

        ]

    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mae")

    model.summary()

    return model


np.random.seed(1291295)

km = KMeans(n_clusters=2)

# # centers = km.fit_transform(x_train_encodings)

km.fit_transform(spect_db.transpose())

cluster_ids = pd.Series(km.labels_).rolling(20*4, center=True).quantile(0.9).fillna(0)



print(np.unique(km.labels_, return_counts=True))



fig, ax = plt.subplots(3,1,figsize=(24, 5), sharex=True)

plt.imshow(spect_db)

ax[0].plot(km.labels_)

ax[1].plot(cluster_ids)



plt.show()
y_train = cluster_ids[:len(x_train)]
def get_classifier():

    clf = RandomForestClassifier(n_estimators=10)

    return clf

clf = get_classifier()

clf.fit(x_train.reshape(len(x_train), -1), y_train.astype(int))
clf.score(x_train.reshape(len(x_train), -1), y_train.astype(int))
spect_db2 = calc_stft_feats(telemetry2)

# Create sequences from test values.

x_test = create_sequences(spect_db2.transpose()[:,feats_touse_index])

print("Test input shape: ", x_test.shape)
x_test_pred = clf.predict(x_test.reshape(len(x_test), -1))

x_test_pred = np.abs(x_test_pred-1)



x_test_pred_smooth = pd.Series(x_test_pred).rolling(20*4, center=True).quantile(0.9).fillna(0)
fig, ax = plt.subplots(figsize=(28,12))

telemetry2[:12_000].plot(legend=False, ax=ax)

plt.plot(x_test_pred_smooth[:12_000]*80, "-r")

plt.show()