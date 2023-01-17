!cd /kaggle/input/audio2numpy

!mkdir audio

!cp -r /kaggle/input/audio2numpy/* ./audio/

from audio.audio2numpy import open_audio

!pip install python_speech_features 2>/dev/null 1>/dev/null
import os

import gc

from tqdm import tqdm_notebook

import warnings

warnings.filterwarnings("ignore")

import random





import numpy as np

import pandas as pd



from PIL import Image

import wave

from scipy.io import wavfile

import librosa

from librosa.feature import melspectrogram





import tensorflow.keras.layers as L

from tensorflow.keras.models import Model

import tensorflow as tf 

from sklearn.metrics import accuracy_score

from joblib import Parallel, delayed

from sklearn.preprocessing import LabelEncoder



from sklearn.utils import shuffle

from sklearn.utils import class_weight



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from python_speech_features import mfcc,logfbank
 

train = pd.read_csv("/kaggle/input/birdsong-recognition/train.csv")

train = train.query("rating>=4")



birds_count = {}

for bird_species, count in zip(train.ebird_code.unique(), train.groupby("ebird_code")["ebird_code"].count().values):

    birds_count[bird_species] = count

most_represented_birds = [key for key,value in birds_count.items() if value == 100]



train_nocall = train.loc[~train.ebird_code.isin(most_represented_birds)] 



train = train.loc[train.ebird_code.isin(most_represented_birds)]
train["bird_name"] = train.ebird_code

train_nocall["bird_name"] = train_nocall.ebird_code

train_nocall = train_nocall.sample(1000)

train_nocall.ebird_code = "nocall"

train = pd.concat([train,train_nocall])

train.reset_index(inplace=True)
def c_fft(y,r):

    n=len(y)

    f=np.fft.rfftfreq(n,1/r)

    f2 = abs(np.fft.rfft(y)/n)

    return (f2,f)





def envelope(y,rate,th):

    mask = []

    

    y = pd.Series(y).apply(np.abs)

    

    y_mean = y.rolling(window = int(rate/10),min_periods = 1, center = True).mean()

    for m in y_mean:

        

        if m > th :

            mask.append(True)

        else :

            mask.append(False)

        

    return mask

sig = {}

fft = {}

fbank = {}

mfcc_s = {}



for i in most_represented_birds:

    w_fn = train.loc[train.ebird_code == i,"filename"].values

    s,r = librosa.load("/kaggle/input/birdsong-recognition/train_audio/"+i+"/"+w_fn[0],sr=44100)

    mask=envelope(s,r,0.005)

    s = s[mask]

    sig[i] = s

    fft[i] = c_fft(s,r)

    fbank[i] = logfbank(s[:r],r,nfilt = 26,nfft=1103).T

    mfcc_s[i] = mfcc(s[:r],r,numcep=13,nfilt=26,nfft=1103).T

def plot_sig(sig):

    fig,axes = plt.subplots(nrows=10,ncols=5,sharex=False,sharey=True,figsize=(20,18))

    i=0

    for x in range(10):

        for y in range(5):

            if x == 9 and y == 4:

                pass

            else :

                axes[x,y].set_title(list(sig.keys())[i])

                axes[x,y].plot(list(sig.values())[i])

                axes[x,y].get_xaxis().set_visible(False)

                axes[x,y].get_yaxis().set_visible(False)

                i += 1  

                

plot_sig(sig)

plt.show()
def plot_fft(fft):

    fig,axes = plt.subplots(nrows=10,ncols=5,sharex=False,sharey=True,figsize=(20,18))

    i=0

    for x in range(10):

        for y in range(5):

            if x == 9 and y == 4:

                pass

            else :

                data = list(fft.values())[i]

                Y ,f = data[0], data[1]

                axes[x,y].set_title(list(fft.keys())[i])

                axes[x,y].plot(f,Y)

                axes[x,y].get_xaxis().set_visible(False)

                axes[x,y].get_yaxis().set_visible(False)

                i += 1 



plot_fft(fft)

plt.show()
def plot_fbank(fbank):

    fig,axes = plt.subplots(nrows=10,ncols=5,sharex=False,sharey=True,figsize=(20,18))

    i=0

    for x in range(10):

        for y in range(5):

            if x == 9 and y == 4:

                pass

            else :

                axes[x,y].set_title(list(fbank.keys())[i])

                axes[x,y].imshow(list(fbank.values())[i],cmap="hot",interpolation="nearest")

                axes[x,y].get_xaxis().set_visible(False)

                axes[x,y].get_yaxis().set_visible(False)

                i += 1  



plot_fbank(fbank)

plt.show()

def plot_mfcc_s(mfcc):

    fig,axes = plt.subplots(nrows=10,ncols=5,sharex=False,sharey=True,figsize=(20,18))

    i=0

    for x in range(10):

        for y in range(5):

            if x == 9 and y == 4:

                pass

            else :

                axes[x,y].set_title(list(fbank.keys())[i])

                axes[x,y].imshow(list(fbank.values())[i],cmap="hot",interpolation="nearest")

                axes[x,y].get_xaxis().set_visible(False)

                axes[x,y].get_yaxis().set_visible(False)

                i += 1  



plot_mfcc_s(mfcc_s)

plt.show()
gc.collect()
NUM_BINS = 1000

MAX_LEN = 10000
def audio_read(path):

    recording, sr = open_audio(path)

    

    if recording.shape[0] != recording.size:

        return recording.mean(axis=1) 

    else:

        return recording
def tokenize(path, NUM_BINS = NUM_BINS,MAX_LEN=MAX_LEN):

    signal = np.resize(audio_read(path), (MAX_LEN,))

    signal_bins = np.linspace(signal.min(), signal.max(), NUM_BINS + 1)

    signal = np.digitize(signal, bins=signal_bins) - 1 

    signal = np.minimum(signal, NUM_BINS - 1)

    return signal
train_audio = []

directory = "/kaggle/input/birdsong-recognition/train_audio/"

for idx, row in train.iterrows():

    train_audio.append(directory+row.bird_name+'/'+row.filename)

    

train_audio = np.array(train_audio)
%%time

X = np.array([Parallel(n_jobs=4)(delayed(tokenize)(filename) for filename in tqdm_notebook(train_audio))])[0]
def build_model(MAX_LEN = MAX_LEN, NUM_BINS = NUM_BINS):

    

    ids = L.Input((MAX_LEN,))

    x = L.Embedding(NUM_BINS,132,input_length=MAX_LEN)(ids)

    x = L.Bidirectional(L.LSTM(132, return_sequences=True))(x)

    x = L.Flatten()(x)

    x = L.Dense(len(most_represented_birds)+1,activation='softmax')(x)



    model = Model(inputs=[ids], outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics = 'accuracy')



    return model
model = build_model()

print(model.summary())
le = LabelEncoder()

seed = 43

le.fit(train['ebird_code'])

Y = pd.get_dummies(train['ebird_code']).values

le.inverse_transform([1])
X.shape,Y.shape
gc.collect()
class_weights = class_weight.compute_class_weight('balanced',np.unique(train.ebird_code),train.ebird_code)
with tf.device('/gpu:0'):

    model.fit(X,Y,epochs=5)
test = pd.read_csv('/kaggle/input/birdsong-recognition/test.csv')

TEST_FOLDER = '../input/birdsong-recognition/test_audio/'

try:

    preds = []

    for index, row in test.iterrows():

        # Get test row information

        site = row['site']

        start_time = row['seconds'] - 5

        row_id = row['row_id']

        audio_id = row['audio_id']



        if site == 'site_1' or site == 'site_2':

            x = np.array([tokenize(TEST_FOLDER + audio_id + '.mp3',start_time)])

        else:

            x = np.array([tokenize(TEST_FOLDER + audio_id + '.mp3')])

        

        pred = le.inverse_transform(np.argmax(model.predict(x),axis=1))[0]

        preds.append([row_id, pred])

except:

     preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')

        

preds = pd.DataFrame(preds, columns=['row_id', 'birds'])

preds.to_csv('submission.csv', index = False)