!pip install -U -q kaggle

!mkdir -p ~/.kaggle
from google.colab import files

files.upload()
!cp kaggle.json ~/.kaggle/
!kaggle datasets list
!kaggle datasets download -d pavansanagapati/urban-sound-classification
import zipfile       
sound_urban = zipfile.ZipFile('/content/urban-sound-classification.zip') #-Pego o arquivo para descopactar.

sound_urban.extractall('/content')#----------------------------------------Local que será descopactado.

 

sound_urban.close()#-------------------------------------------------------Fechando arquivo.
train_sound = zipfile.ZipFile('/content/train.zip') #-Pego o arquivo para descopactar.

train_sound.extractall('/content')#----------------------------------------Local que será descopactado.

 

train_sound.close()#-------------------------------------------------------Fechando arquivo.
test_sound = zipfile.ZipFile('/content/test.zip') #-Pego o arquivo para descopactar.

test_sound.extractall('/content')#----------------------------------------Local que será descopactado.

 

test_sound.close()#-------------------------------------------------------Fechando arquivo.
!pip install soundfile

!mkdir /content/espc
import os



import numpy as np

import pandas as pd



import librosa

import librosa.display

import soundfile as sf # librosa fails when reading files on Kaggle.

from IPython.display import Image

import IPython.display as ipd 



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from glob import glob



from sklearn.model_selection import train_test_split#-Dividir dados em treino e teste.

from sklearn.preprocessing import StandardScaler#-----Padroniza os dados.

from sklearn.decomposition import PCA#----------------

from sklearn.neighbors import KNeighborsClassifier#---Modelo de classificação com os vizinhos mais proximos.

from sklearn.model_selection import GridSearchCV#-----Grid.

from sklearn.metrics import confusion_matrix#---------Retorna a matrix de confusão.



from sklearn.preprocessing import minmax_scale



from google.colab import drive
audio_path = '/content/Train/1006.wav'

x, sr = librosa.load(audio_path)

print(type(x),type(sr),'\n')

print('x: ', x)

print('\nO default de sr:', sr)
ipd.Audio (audio_path)
#display waveform 

plt.figure (figsize = (14, 5)) 

librosa.display.waveplot(x,sr=sr);#-Plota a onda

#---------------------------------- Em y se encontra a amplitude do audio e em x o tempo.

plt.grid()#-------------------------Coloca grade no plot.
plt.figure(figsize=(12, 5))#-Gero uma figura e defino sua escala.

plt.plot(x[2000:2100]);#-----Zoom em uma parte da plotagem de onda.

plt.grid()#------------------Plota uma grade.
zero_crossings = librosa.zero_crossings (x [2000: 2100], pad = False) 

print (sum (zero_crossings))
#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

spectral_centroids.shape #retornará uma matriz com colunas iguais a um número de quadros presentes em sua amostra.



# Computando a variável de tempo para visualização

frames = range(len(spectral_centroids))

t = librosa.frames_to_time(frames)#converte frames em times-> frame[i] == time[i]



# Normalizando o centróide espectral para visualização

def normalize(x, axis=0):

    return minmax_scale(x, axis=axis)



#Plotando a centroids na forma de onda

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_centroids), color='r');



print(f'Centroids Shape: {spectral_centroids.shape}')

print(f'Primeiras 3 centroids: {spectral_centroids[:3]}')
X_espc = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X_espc))

plt.figure(figsize=(14, 5))

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 



#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')

plt.colorbar();
plt.figure(figsize=(10,10))

img=mpimg.imread('/content/36c3b4cb-98df-4140-86de-aa964cb8b7ab.jfif')

imgplot = plt.imshow(img)
plt.figure(figsize=(10,10))

img=mpimg.imread('/content/49987cfc-e3a1-477f-a32b-8384c560c961.jfif')

imgplot = plt.imshow(img)
plt.figure(figsize=(10,10))

img=mpimg.imread('/content/mffc_fluxo.PNG')

imgplot = plt.imshow(img)

train_id = pd.read_csv('/content/train.csv')#--Arquivo csv com id dos audioe de treino.

test_id  = pd.read_csv('/content/test.csv')#---Arquivo csv com id dos audioe de teste==
def mean_mfccs(x):

    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]



def parse_audio(x):

    return x.flatten('F')[:x.shape[0]] 



def get_audios():

    train_path = "/content/Train/"

    train_file_names = os.listdir(train_path)

    train_file_names.sort(key=lambda x: int(x.partition('.')[0]))

    

    samples = []

    for file_name in train_file_names:

        x, sr = sf.read(train_path + file_name, always_2d=True)

        #x, sr = librosa.load(audio_path)

        x = parse_audio(x)

        samples.append(mean_mfccs(x))

        

    return np.array(samples)



def get_samples():

    df = pd.read_csv('/content/train.csv')

    return get_audios(), df['Class'].values

  

#Função que cria os espectogramas

def get_spectrogram(filename,name):

    plt.interactive(False)

    clip, sample_rate = librosa.load(filename, sr=None)

    fig = plt.figure(figsize=[0.72,0.72])

    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    ax.set_frame_on(False)

    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    filename  = '/content/espc/' + name + '.jpg'

    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)

    plt.close()    

    fig.clf()

    plt.close(fig)

    plt.close('all')

    del filename,name,clip,sample_rate,fig,ax,S
X, Y = get_samples()
x_train, x_test, y_train, y_test = train_test_split(X, Y)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
Data_dir=np.array(glob("/content/Train/*"))

i=1500

for file in Data_dir[i:i+1500]:

    filename,name = file,file.split('/')[-1].split('.')[0]

    get_spectrogram(filename,name)
Data_dir.shape
scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_test)
grid_params = {

    'n_neighbors': [3, 5, 7, 9, 11, 15],

    'weights': ['uniform', 'distance'],

    'metric': ['euclidean', 'manhattan']

}



model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1);

model.fit(x_train_scaled, y_train);
print(f'Model Score: {model.score(x_test_scaled, y_test)}')



y_predict = model.predict(x_test_scaled)

print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')


!pip install pydub

from pydub import AudioSegment

sound = AudioSegment.from_mp3("/content/Pistola22cal.mp3")

sound.export("/content/pred.wav", format="wav");
we
def mean_mfccs_2(x):

    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]



def parse_audio_2(x):

    return x.flatten('F')[:x.shape[0]] 



def get_audios_2():

    train_path = "/content/Train/"

    train_file_names = os.listdir(train_path)

    train_file_names.sort(key=lambda x: int(x.partition('.')[0]))

    



samples = []

    

x, sr = sf.read('/content/pred.wav', always_2d=True)

#x, sr = librosa.load(audio_path)

x = parse_audio_2(x)

samples.append(mean_mfccs_2(x))

        

mfcc =  np.array(samples)



mfcc_scaler = scaler.transform(mfcc)
model.predict(mfcc_scaler)
ipd.Audio ('/content/pred.wav')
X_espc = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X_espc))

plt.figure(figsize=(14, 5))

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 



#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')

plt.colorbar();
train_id