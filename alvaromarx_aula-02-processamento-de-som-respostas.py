# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import librosa

import librosa.display

import soundfile as sf

import seaborn as sns

import matplotlib.pyplot as plt

import IPython.display as ipd

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        break

# Any results you write to the current directory are saved as output.
hiphop, sr_hiphop = sf.read("/kaggle/input/gtzan-genre-collection/genres/hiphop/hiphop.00049.au")

classic, sr_classic = sf.read("/kaggle/input/gtzan-genre-collection/genres/classical/classical.00007.au")
ipd.Audio(hiphop,rate=sr_hiphop)
ipd.Audio(classic,rate=sr_classic)
plt.plot(classic)

plt.show()

plt.plot(hiphop)

plt.show()
second_hiphop  = hiphop[:sr_hiphop]

second_classic  = classic[:sr_classic]
plt.plot(second_hiphop)

plt.show()

plt.plot(second_classic)

plt.show()
rms_hiphop = librosa.feature.rms(y=second_hiphop)

rms_classic = librosa.feature.rms(y=second_classic)
print(f"HIP HOP: {rms_hiphop.shape}")

print(f"CLASSIC: {rms_classic.shape}")
plt.plot(rms_hiphop[0])

plt.plot(rms_classic[0])

plt.show()
centroid_hiphop = librosa.feature.spectral_centroid(second_hiphop)

centroid_classic = librosa.feature.spectral_centroid(second_classic)
plt.plot(centroid_hiphop[0])

plt.plot(centroid_classic[0])

plt.show()
flatness_hiphop = librosa.feature.spectral_flatness(second_hiphop)

flatness_classic = librosa.feature.spectral_flatness(second_classic)
plt.plot(flatness_hiphop[0])

plt.plot(flatness_classic[0])

plt.show()
mfcc_hiphop = librosa.feature.mfcc(y=second_hiphop, sr=sr_hiphop)

mfcc_classic = librosa.feature.mfcc(y=second_classic, sr=sr_classic)

print(f"hiphop: {mfcc_hiphop.shape}")

print(f"classic: {mfcc_classic.shape}")
librosa.display.specshow(mfcc_hiphop,sr=sr_hiphop, x_axis='time')

plt.show()

librosa.display.specshow(mfcc_classic, sr=sr_classic, x_axis='time')

plt.show()
f1 = librosa.feature.spectral_centroid

f2 = librosa.feature.spectral_flatness
fts = np.zeros((1,3))

for window in np.array_split(hiphop, 60):

    f = np.hstack([f1(y=window).mean(), f2(y=window).mean(), 0])

    fts = np.vstack([fts, f])

for window in np.array_split(classic, 60):

    f = np.hstack([f1(y=window).mean(), f2(y=window).mean(), 1])

    fts = np.vstack([fts, f])

fts = fts[1:,:]
df = pd.DataFrame(data=fts)

df.head()
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC



X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values



scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)



model = SVC(random_state=42)

print("treinando modelo...")

model.fit(X_scaled, y)



genres = ['hiphop','classical']

_fts = np.zeros((1,3))

print("carregando dados de teste...")

for genre in genres:

    path = os.path.join("/kaggle/input/gtzan-genre-collection/genres/",genre)

    for sound in os.listdir(path):

        soundfile = os.path.join(path, sound)

        s, sr = sf.read(soundfile)

        f = np.hstack([f1(y=s).mean(), f2(y=s).mean(), genres.index(genre)])

        _fts = np.vstack([_fts, f])

_fts = _fts[1:,:]

print("tratando dados de teste...")

X_test, y_test = _fts[:,:-1], _fts[:,-1]

X_test = scaler.transform(X_test)

print("medindo acurácia...")

print(f"acurácia: {100*model.score(X_test, y_test)}%")
y_pred = model.predict(X_test)

sns.scatterplot(X_test[:,0],X_test[:,1],hue=[genres[i] for i in y_test.astype("int")])

plt.title("Dados atuais")

plt.xlabel(f1.__name__)

plt.ylabel(f2.__name__)

plt.show()

sns.scatterplot(X_test[:,0],X_test[:,1],hue=[genres[i] for i in y_pred.astype("int")])

plt.title("Dados previstos pelo modelo")

plt.xlabel(f1.__name__)

plt.ylabel(f2.__name__)

plt.show()