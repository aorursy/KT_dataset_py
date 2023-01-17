import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
sns.set(color_codes=True)
path = os.path.join(os.getcwd(), "genres")
print(path)
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
for genre in genres:
    folder = os.path.join(path, genre) # selects each folder using the genre name from the list of genres
    for file in os.listdir(folder): # iterates over each file in the genre sub-folder
        filename = os.path.join(folder, file)
        y, sr = librosa.load(filename) # y is auido waveform, sr is sampling rate (22050 default)
        librosa.display.waveplot(y, sr=sr)
        break
    break

# The two breaks are introduced to break the for loop beforehand as we don't
# necessarily need to generate 1000 waveplots, hence generating only one waveplot,
# which is the last track of rock (rock.00099)
    
Y = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(Y))
plt.figure(figsize = (14, 5))
librosa.display.specshow(Xdb, sr = sr, x_axis = "time", y_axis = "hz")
plt.colorbar()
librosa.display.specshow(Xdb, sr = sr, x_axis = "time", y_axis = "log")
plt.colorbar()
# Testing the features function beforehand
mel = librosa.filters.mel(sr = sr, n_fft = 2048, n_mels = 128)
# Visualising our example melspectrogram
librosa.display.specshow(spec_db, sr = sr, hop_length = 512, x_axis = "time", y_axis = "mel")
# Creation of the 1000 mel spectrograms
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
for genre in genres:
    folder = os.path.join(path, genre) # selects each folder using the genre name from the list of genres
    for file in os.listdir(folder): # iterates over each file in the genre sub-folder
        filename = os.path.join(folder, file)
        y, sr = librosa.load(filename) # y is auido waveform, sr is sampling rate (22050 default)
        spec = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, hop_length = 512, n_mels = 128)
        spec_db = librosa.power_to_db(spec, ref = np.max)
        spec_show = librosa.display.specshow(spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.axis('off');
        plt.savefig(f'genres/{genre}/{file[:-3].replace(".", "")}.png')
        plt.clf()
header = 'filename chroma_stft chroma_sd cqt cqt_sd tonnetz tonnetz_std rmse rmse_sd spectral_centroid speccent_sd spectral_bandwidth specband_sd rolloff rolloff_sd zero_crossing_rate zrc_sd'
for i in range(1, 21):
    header += f' mfcc{i} mfcc_sd{i}'
header += ' label'
header = header.split()
import csv

file = open('data.csv', 'w', newline = '')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename  in os.listdir(f'./genres/{g}'):
        songname = f'./genres/{g}/{filename}'
        if "wav" not in songname:  # to have the function move on even after meeting our spectrogram images(png files)
            continue
        y, sr = librosa.load(songname, mono  = True, duration = 30)
        chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr) # measures the energy in each pitch class
        cqt = librosa.feature.chroma_cqt(y = y, sr = sr) # measures the energy in each pitch, which is basically the chrome frequency.
        spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr) #Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins, from which the mean (centroid) is extracted per frame.
        tonnetz = librosa.feature.tonnetz(y = y, sr = sr)
        rmse = librosa.feature.rmse(y = y) # computes the root mean square of energy within each frame
        spec_bw = librosa.feature.spectral_bandwidth(y = y , sr = sr) 
        rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
        zcr = librosa.feature.zero_crossing_rate(y) #computes the zero crossing rate of an audio time series
        mfcc = librosa.feature.mfcc(y = y, sr = sr) # mel frequency cepstral coefficients
        to_append = f'{filename} {np.mean(chroma_stft)} {np.std(chroma_stft)} {np.mean(cqt)} {np.std(cqt)} {np.mean(tonnetz)} {np.std(tonnetz)} {np.mean(rmse)} {np.std(rmse)} {np.mean(spec_cent)} {np.std(spec_cent)} {np.mean(spec_bw)} {np.std(spec_bw)} {np.mean(rolloff)} {np.std(rolloff)} {np.mean(zcr)} {np.std(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)} {np.std(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline = '')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
data = pd.read_csv("data.csv")
data.head()
data.shape
#data.isnull().sum()
data.skew(axis = 0, skipna = True)
#data.dtypes
list_remove = ['rmse', 'rmse_sd', 'spectral_bandwidth', 'specband_sd']
data1 = data.drop(data.loc[:, 'mfcc6' : 'mfcc_sd20'].columns, axis = 1)
data1 = data1.drop(list_remove, axis = 1)
data1.head()
fig, ax = plt.subplots(figsize = (10, 6))
sns.boxplot(x = data1['label'], y = data1['cqt'])
fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(data1['cqt'], data1['label'])
ax.set_xlabel('Constant Q Transform')
ax.set_ylabel('Genres')
plt.show()
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_histogram, geom_boxplot, geom_line
ggplot(data1) + geom_histogram(aes(x = 'cqt'), color = 'blue') + facet_wrap('~label')
plt.figure(figsize = (20, 10))
sns.distplot(a=data1[data1['label'] == 'rock']['cqt'], color = 'red', kde = True, bins = 20, hist = False, label='rock')
sns.distplot(a=data1[data1['label'] == 'classical']['cqt'], color = 'gold', kde = True, bins = 20, hist = False, label='classical')
sns.distplot(a=data1[data1['label'] == 'hiphop']['cqt'], color = 'purple', kde = True, bins = 20, hist = False, label='hiphop')
sns.distplot(a=data1[data1['label'] == 'metal']['cqt'], color = 'silver', kde = True, bins = 20, hist = False, label='metal')
sns.distplot(a=data1[data1['label'] == 'blues']['cqt'], color = 'blue', kde = True, bins = 20, hist = False, label='blues')
sns.distplot(a=data1[data1['label'] == 'reggae']['cqt'], color = 'maroon', kde = True, bins = 20, hist = False, label='reggae')
sns.distplot(a=data1[data1['label'] == 'jazz']['cqt'], color = 'navy', kde = True, bins = 20, hist = False, label='jazz')
sns.distplot(a=data1[data1['label'] == 'country']['cqt'], color = 'green', kde = True, bins = 20, hist = False, label='country')
sns.distplot(a=data1[data1['label'] == 'disco']['cqt'], color = 'pink', kde = True, bins = 20, hist = False, label='disco')
sns.distplot(a=data1[data1['label'] == 'pop']['cqt'], color = 'black', kde = True, bins = 20, hist = False, label='metal')
plt.legend(fontsize = 15)
plt.title('Constant Q Transform based on Genres', fontsize = 20)
plt.figure(figsize = (20, 10))
sns.distplot(a=data1[data1['label'] == 'rock']['zero_crossing_rate'], color = 'red', kde = True, bins = 20, hist = False, label='rock')
sns.distplot(a=data1[data1['label'] == 'classical']['zero_crossing_rate'], color = 'gold', kde = True, bins = 20, hist = False, label='classical')
sns.distplot(a=data1[data1['label'] == 'hiphop']['zero_crossing_rate'], color = 'purple', kde = True, bins = 20, hist = False, label='hiphop')
sns.distplot(a=data1[data1['label'] == 'metal']['zero_crossing_rate'], color = 'silver', kde = True, bins = 20, hist = False, label='metal')
sns.distplot(a=data1[data1['label'] == 'blues']['zero_crossing_rate'], color = 'blue', kde = True, bins = 20, hist = False, label='blues')
sns.distplot(a=data1[data1['label'] == 'reggae']['zero_crossing_rate'], color = 'maroon', kde = True, bins = 20, hist = False, label='reggae')
sns.distplot(a=data1[data1['label'] == 'jazz']['zero_crossing_rate'], color = 'navy', kde = True, bins = 20, hist = False, label='jazz')
sns.distplot(a=data1[data1['label'] == 'country']['zero_crossing_rate'], color = 'green', kde = True, bins = 20, hist = False, label='country')
sns.distplot(a=data1[data1['label'] == 'disco']['zero_crossing_rate'], color = 'pink', kde = True, bins = 20, hist = False, label='disco')
sns.distplot(a=data1[data1['label'] == 'pop']['zero_crossing_rate'], color = 'black', kde = True, bins = 20, hist = False, label='mteal')
plt.legend(fontsize = 15)
plt.title('Zero Crossing Rate', fontsize = 20)
plt.figure(figsize = (20, 10))
sns.distplot(a=data1[data1['label'] == 'rock']['tonnetz'], color = 'red', kde = True, bins = 20, hist = False, label='rock')
sns.distplot(a=data1[data1['label'] == 'classical']['tonnetz'], color = 'gold', kde = True, bins = 20, hist = False, label='classical')
sns.distplot(a=data1[data1['label'] == 'hiphop']['tonnetz'], color = 'purple', kde = True, bins = 20, hist = False, label='hiphop')
sns.distplot(a=data1[data1['label'] == 'metal']['tonnetz'], color = 'silver', kde = True, bins = 20, hist = False, label='metal')
sns.distplot(a=data1[data1['label'] == 'blues']['tonnetz'], color = 'blue', kde = True, bins = 20, hist = False, label='blues')
sns.distplot(a=data1[data1['label'] == 'reggae']['tonnetz'], color = 'maroon', kde = True, bins = 20, hist = False, label='reggae')
sns.distplot(a=data1[data1['label'] == 'jazz']['tonnetz'], color = 'navy', kde = True, bins = 20, hist = False, label='jazz')
sns.distplot(a=data1[data1['label'] == 'country']['tonnetz'], color = 'green', kde = True, bins = 20, hist = False, label='country')
sns.distplot(a=data1[data1['label'] == 'disco']['tonnetz'], color = 'pink', kde = True, bins = 20, hist = False, label='disco')
sns.distplot(a=data1[data1['label'] == 'pop']['tonnetz'], color = 'black', kde = True, bins = 20, hist = False, label='mteal')
plt.legend(fontsize = 15)
plt.title("Tonnetz based on Genres", fontsize = 20)
c = data1.corr()
plt.figure(figsize = (20, 10))
c = data1.corr()
sns.heatmap(c, annot = True)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data1.iloc[:, 2:-1], dtype = 'float'))

encoder = LabelEncoder()
genre_col = data1.iloc[:, -1]
y = encoder.fit_transform(genre_col)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_pred

confm = confusion_matrix(y_test, y_pred)
sns.heatmap(confm, annot = True, fmt = "d", linewidths = 1, cmap = "Blues", 
           xticklabels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'],
           yticklabels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])

total = y_test.shape[0]
correct = (y_pred == y_test).sum()
accuracy = (correct / total) * 100

print('Accuracy', round(accuracy, 2))
lreg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
lreg.fit(X_train, y_train)

y_pred = lreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt  ="d", linewidths = 1, cmap = "Blues", 
           xticklabels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'],
           yticklabels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])

totla = y_test.shape[0]
correct = (y_pred == y_test).sum()
accuracy = (correct / total) * 100

print("Accuracy", round(accuracy, 2))
# Confusion matrix between a track of a genre and another

import sklearn
files = ['C:/Users/40195086/genres/blues//blues.00000.wav', "C:/Users/40195086/genres/rock/rock.00099.wav"]
output = []
for file in files:
    y, sr = librosa.load(file)
    cqt = librosa.feature.chroma_cqt(y = y, sr = sr)
    output.append(cqt)
    
a = output[0]
b = output[1]

print(a.shape, b.shape)

sklearn.metrics.pairwise.cosine_similarity(a, b)[0][0] #cosine similarity is a metric used to measure how similar the objects are irrespective of their size. 

cosine_similarity(a, b)[0][0]
classical = data1[data1['label'] == 'classical']
c = classical[['chroma_stft','cqt','spectral_centroid', 'rolloff', 'zero_crossing_rate']].corr()
sns.heatmap(c, annot = True)
c
