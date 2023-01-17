# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly as pl

import plotly.graph_objects as go

import librosa
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        
pd.set_option("display.max_rows", 999, "display.max_columns", 999)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_file = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
test_file = pd.read_csv('/kaggle/input/birdsong-recognition/test.csv')
train_file
train_file.info()
train_file.nunique()
train_file.describe()
x = train_file['ebird_code'].unique()

print("Total Birds in the data set: ", len(x))

x
import requests 

Bird_name = input('Type the bird name: ')
print(Bird_name)

link = "https://ebird.org/species/{0}#".format(Bird_name)
from IPython.display import IFrame
IFrame(link, width=1000, height=700) 
import plotly.express as px
fig = px.histogram(train_file, x="duration")
fig.show()
fig = px.histogram(train_file, x="duration", y="species", histfunc= 'avg').update_yaxes(categoryorder="total descending")
fig.show()
def dateprocessing(row):
    year = row.split('-')[0]
    month = row.split('-')[1]
    date = row.split('-')[2]
    return year, month, date
#train_file = train_file['date'].apply(dateprocessing)
train_file['year'],  train_file['month'], train_file['day'] = zip(*train_file['date'].apply(dateprocessing))
import plotly.express as px
fig = px.histogram(train_file, x="month")
fig.show()
import plotly.express as px
fig = px.histogram(train_file, x="day")
fig.show()
train_file
pitch= train_file["pitch"].value_counts().sort_values()

pitch.plot.barh()
x = train_file['species'].value_counts()[train_file['species'].value_counts() < 100]
fig = go.FigureWidget(data=go.Bar(y=x))
fig
train_file["species"].value_counts(ascending= True)[:30].sort_values()
import random 
import os


path ='/kaggle/input/birdsong-recognition/train_audio/{0}/'.format(Bird_name)
files = os.listdir(path)
index = random.randrange(0, len(files))
rndm_file = files[index]


input_audio = '/kaggle/input/birdsong-recognition/train_audio/{0}/{1}'.format(Bird_name,rndm_file)
data, sr = librosa.load(input_audio, sr = 44100) 
print('Audio loaded: ', data, sr)

print('\nlength of the numpy array,' , len(data))
#data = librosa.effects.trim(data)
time = np.arange(0, len(data))/ sr
time
import IPython.display as ipd
ipd.Audio(input_audio)
fig, ax = plt.subplots()
ax.plot(time, data)
ax.set(xlabel='Time', ylabel = 'Amplitude')
import librosa.display
plt.figure(figsize=(30, 4))
librosa.display.waveplot(data, sr=sr)
X = librosa.stft(data,)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
spectral_centroids.shape
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')
