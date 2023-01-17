import os



import numpy as np

import pandas as pd



import librosa

import librosa.display

import soundfile as sf # librosa fails when reading files on Kaggle.



import matplotlib.pyplot as plt

import IPython.display as ipd



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix
audio_path = '../input/train/Train/2.wav'

ipd.Audio(audio_path)
# Extract the audio data (x) and the sample rate (sr).

x, sr = librosa.load(audio_path)



# Plot the sample.

plt.figure(figsize=(12, 5))

librosa.display.waveplot(x, sr=sr)

plt.show()
plt.figure(figsize=(12, 5))

plt.plot(x[1000:1100]) # Zoom-in for seeing the example.

plt.grid()



n_crossings = librosa.zero_crossings(x[1000:1100], pad=False)

print(f'Number of crosses: {sum(n_crossings)}')
centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]



print(f'Centroids Shape: {centroids.shape}')

print(f'First 3 centroids: {centroids[:3]}')
mfccs = librosa.feature.mfcc(x, sr=sr)

print(f'MFFCs shape: {mfccs.shape}')

print(f'First mffcs: {mfccs[0, :5]}')



# We can even display an spectogram of the mfccs.

librosa.display.specshow(mfccs, sr=sr, x_axis='time')

plt.show()
def mean_mfccs(x):

    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]



def parse_audio(x):

    return x.flatten('F')[:x.shape[0]] 



def get_audios():

    train_path = "../input/train/Train/"

    train_file_names = os.listdir(train_path)

    train_file_names.sort(key=lambda x: int(x.partition('.')[0]))

    

    samples = []

    for file_name in train_file_names:

        x, sr = sf.read(train_path + file_name, always_2d=True)

        x = parse_audio(x)

        samples.append(mean_mfccs(x))

        

    return np.array(samples)



def get_samples():

    df = pd.read_csv('../input/train.csv')

    return get_audios(), df['Class'].values
X, Y = get_samples()



# Since the data manufacturer doesn't provide the labels for the test audios,

# we will have do the split for the labeled data.

x_train, x_test, y_train, y_test = train_test_split(X, Y)
print(f'Shape: {x_train.shape}')

print(f'Observation: \n{x_train[0]}')

print(f'Labels: {y_train[:5]}')
scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_test)



pca = PCA().fit(x_train_scaled)



plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)')

plt.show()
grid_params = {

    'n_neighbors': [3, 5, 7, 9, 11, 15],

    'weights': ['uniform', 'distance'],

    'metric': ['euclidean', 'manhattan']

}



model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)

model.fit(x_train_scaled, y_train)
print(f'Model Score: {model.score(x_test_scaled, y_test)}')



y_predict = model.predict(x_test_scaled)

print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')