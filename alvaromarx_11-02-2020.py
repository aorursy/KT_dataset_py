# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import soundfile as sf

import librosa.feature as libft

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sound, samplerate = sf.read("/kaggle/input/florestsoundtavares/ZOOM0041.WAV")

samplerate


def window(sound, samplerate, step, over):



    st = int(step*samplerate)

    of = int(over*samplerate)



    print(st, of)

    x = [sound[int(i):int(i+st)] for i in np.arange(0,sound.shape[0],of)]

    for i in range(len(x)-1, 0, -1):

        if x[i].shape[0] < st:

            x[i] = np.concatenate((x[i], np.zeros((st-x[i].shape[0],2))))

        else:

            break



    return np.vstack(x).reshape(len(x),st,2)
sound_rs = window(sound, samplerate, 1, 0.5)

sound_rs_ch1 = sound_rs[:-2,:,0]

sound_rs_ch2 = sound_rs[:,:,1]
def get_features_from_window(window, samplerate):

    # MFCC, spectal centroid, spectral flatness and spectral rolloff

    # All features with 1st and 2nd diferentials

    mfcc = libft.mfcc(window,sr=samplerate,n_mfcc=12)

    spectral_flatness = libft.spectral_flatness(window)

    spectral_rolloff = libft.spectral_rolloff(window)

    spectral_centroid = libft.spectral_centroid(window)

    delta_mfcc = libft.delta(mfcc)

    delta2_mfcc = libft.delta(mfcc,order=2)

    delta_spectral_flatness = libft.delta(spectral_flatness)

    delta2_spectral_flatness = libft.delta(spectral_flatness,order=2)

    delta_spectral_rolloff = libft.delta(spectral_rolloff)

    delta2_spectral_rolloff = libft.delta(spectral_rolloff,order=2)

    delta_spectral_centroid = libft.delta(spectral_centroid)

    delta2_spectral_centroid = libft.delta(spectral_centroid,order=2)

    feature_array = np.array([mfcc.mean(), np.std(mfcc), delta_mfcc.mean(), np.std(delta_mfcc), delta2_mfcc.mean(), np.std(delta2_mfcc), spectral_flatness.mean(), np.std(spectral_flatness), delta_spectral_flatness.mean(), np.std(delta_spectral_flatness), delta2_spectral_flatness.mean(), np.std(delta2_spectral_flatness), spectral_rolloff.mean(), np.std(spectral_rolloff), delta_spectral_rolloff.mean(), np.std(delta_spectral_rolloff), delta2_spectral_rolloff.mean(), np.std(delta2_spectral_rolloff), spectral_centroid.mean(), np.std(spectral_centroid), delta_spectral_centroid.mean(), np.std(delta_spectral_centroid), delta2_spectral_centroid.mean(), np.std(delta2_spectral_centroid)])

    return feature_array

dataset_ch1_values = np.zeros(24).reshape(1,24)

for window in sound_rs_ch1:

    dataset_ch1_values = np.vstack([dataset_ch1_values, get_features_from_window(np.asfortranarray(window), samplerate)])
columns = ''.join([f'{i}_mean {i}_std delta_{i}_mean delta_{i}_std delta2_{i}_mean delta2_{i}_std ' for i in ['mfcc', 'spectral_flatness','spectral_rolloff','spectral_centroid']]).split()
# dataset_ch1 = pd.read_csv("/kaggle/input/ch1manaus/dataset_ch1.csv")

# dataset_ch1 = dataset_ch1.drop(0)

# dataset_ch1 = dataset_ch1.iloc[:-2]

# dataset_ch1_values = dataset_ch1.values

dataset_ch1 = pd.DataFrame(data=dataset_ch1_values, columns=columns)

dataset_ch1 = dataset_ch1.drop(0)

dataset_ch1.head()
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
scaler = StandardScaler()

scaler.fit(dataset_ch1.values)
transformed_ch1_values = scaler.transform(dataset_ch1.values)

pca = PCA(transformed_ch1_values.shape[1]-1)

pca.fit(transformed_ch1_values)

print(pca.explained_variance_ratio_.sum())

pca_ch1_values = pca.transform(transformed_ch1_values)
from sklearn.cluster import KMeans
kmeans_models = [KMeans(n_clusters=i, init="k-means++",random_state=42) for i in range(1,25)]
for i, model in enumerate(kmeans_models):

    kmeans_models[i].fit(pca_ch1_values)

for i, model in enumerate(kmeans_models):

    plt.plot(i, model.inertia_, 'ro')
MODEL_SEL = 4

model_selected= kmeans_models[MODEL_SEL]

model_selected.inertia_
predicted = model_selected.predict(pca_ch1_values)

predicted.resize(1,pca_ch1_values.shape[0])

!rm -r label_*

for label in range(MODEL_SEL+1):

    if not os.path.exists(f"/kaggle/working/label_{label}"):

        os.mkdir(f"/kaggle/working/label_{label}")

    for enum, s in enumerate(sound_rs_ch1[predicted[0] == label]):

        if not os.path.exists(f"/kaggle/working/label_{label}/sound_{enum}.wav"):

            sf.write(f"/kaggle/working/label_{label}/sound_{enum}.wav", s, 96000)

        if enum > 10:

            break

    
import IPython.display as ipd

import librosa.display as libdis

import librosa

for label in range(MODEL_SEL+1):

    print(f"##### LABEL {label} ######")

    for enum, s in enumerate(sound_rs_ch1[predicted[0] == label]):

        D = librosa.amplitude_to_db(np.abs(librosa.stft(s)), ref=np.max)

        ipd.display(libdis.specshow(D, y_axis='log'))

        plt.show()

        break
import IPython.display as ipd

for label in range(MODEL_SEL+1):

    print(f"##### LABEL {label} ######")

    for enum, s in enumerate(sound_rs_ch1[predicted[0] == label]):

        ipd.display(ipd.Audio(f"/kaggle/working/label_{label}/sound_{enum}.wav"))

        if enum > 5:

            break

plt.plot(range(MODEL_SEL+1),[predicted[0][predicted[0] == i].size for i in range(MODEL_SEL+1)])
import pylab

pylab.scatter(pca_ch1_values[:,0], pca_ch1_values[:,1],c=predicted[0])