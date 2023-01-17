!apt-get install libsndfile1 -y
from glob import glob

import numpy as np

from keras import Sequential

from keras.callbacks import EarlyStopping

from keras.layers import Dense, Dropout

from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler, LabelEncoder

data_path = "/kaggle/input/raw-data/raw_data/train/train/*"



import librosa



def hpss(signal):

    signal = librosa.effects.hpss(signal.astype('float64'))

    return signal[1]



def random_shift(signal):

    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length

    start = int(signal.shape[0] * timeshift_fac)

    if start > 0:

        signal = np.pad(signal,(start,0),mode='constant')[0:signal.shape[0]]

    else:

        signal = np.pad(signal,(0,-start),mode='constant')[0:signal.shape[0]]

    return signal



def add_noise(signal):

    noise_amp = 0.005*np.random.uniform()*np.amax(signal)

    signal = signal.astype('float64') + noise_amp * np.random.normal(size=signal.shape[0])



    return signal



def change_pitch(signal, sample_rate):

    bins_per_octave = 12

    pitch_pm = 2

    pitch_change =  pitch_pm * 2*(np.random.uniform())

    signal = librosa.effects.pitch_shift(signal.astype('float64'),

                                          sample_rate, n_steps=pitch_change,

                                          bins_per_octave=bins_per_octave)

    return signal
def extract_features(file_name, apply_add_noise=False, apply_change_pitch=False, apply_hpss=False, apply_random_shift=False):

    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')



    if apply_random_shift:

        X = random_shift(X)



    if apply_hpss:

        X = hpss(X)



    if change_pitch:

        #data augmentation change pitch

        X = change_pitch(X, sample_rate)



    if add_noise:

        # data agumentation, add random noise

        X = add_noise(X)



    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)



    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft

    stft = np.abs(librosa.stft(X))



    # Computes a chromagram from a waveform or power spectrogram.

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)



    # Computes a mel-scaled spectrogram.

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)



    # Computes spectral contrast

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)



    # Computes the tonal centroid features (tonnetz)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)



    zero_crossing = librosa.feature.zero_crossing_rate(X, pad=False).flatten()



    spectral_centroid = librosa.feature.spectral_centroid(X, sr=sample_rate).flatten()



    spectral_rolloff = librosa.feature.spectral_rolloff(X, sr=sample_rate).flatten()



    rms = librosa.feature.rms(X).flatten()



    poly_features = librosa.feature.poly_features(X, sr=sample_rate).flatten()



    return np.concatenate([mfccs, chroma, mel, contrast, tonnetz, zero_crossing, spectral_centroid, spectral_rolloff, rms, poly_features ])
from tqdm import tqdm

data_dir = np.array(glob(data_path))

features, labels = [], []

for file in tqdm(data_dir):

    

    file_name = file.split("/")[-1].split(".")[0]



    name, label = file_name.split("-")[0], file_name.split("-")[1]

    assert label == '1' or label == '0'





    features.append(extract_features(file))

    labels.append(label)



    features.append(extract_features(file,apply_add_noise=True))

    labels.append(label)



    features.append(extract_features(file, apply_change_pitch=True))

    labels.append(label)



    features.append(extract_features(file, apply_add_noise=True, apply_change_pitch=True))

    labels.append(label)

    

    features.append(extract_features(file, apply_add_noise=True, apply_hpss=True))

    labels.append(label)



    



from sklearn.model_selection import train_test_split

inputs_train, inputs_test, targets_train, targets_test = train_test_split(features, labels, test_size=0.2)


ss = StandardScaler()

X_train = ss.fit_transform(inputs_train)

X_val = ss.transform(inputs_test)





lb = LabelEncoder()

y_train = to_categorical(lb.fit_transform(targets_train))

y_val = to_categorical(lb.fit_transform(targets_test))



input_shape = X_train[0].shape



model = Sequential()



model.add(Dense(457, input_shape=input_shape, activation = 'relu'))

model.add(Dropout(0.2))



model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.2))



model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.5))



model.add(Dense(2, activation = 'softmax'))



model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')



early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

model.summary()
history = model.fit(X_train, y_train,batch_size=500, epochs=50,

                    validation_data=(X_val, y_val),

                    callbacks=[early_stop])
test_path = "/kaggle/input/raw-data/raw_data/test/test/*"

from tqdm import tqdm

test_dir = np.array(glob(test_path))



fout = open("submission.txt", "w")

fout.write("name,label\n")

for file in tqdm(test_dir):

    name = file.split("/")[-1]

    ft = extract_features(file)

    ft = ss.transform([ft])

    pred = model.predict_classes([ft])[0]

    fout.write("{},{}\n".format(name, pred))



fout.close()


