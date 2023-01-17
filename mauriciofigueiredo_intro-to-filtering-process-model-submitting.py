# feature extractoring and preprocessing data

import librosa

import librosa.display

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image

from pathlib import Path

import csv

# Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

#Reports

from sklearn.metrics import classification_report, confusion_matrix



import warnings

warnings.filterwarnings('ignore')

sr=16000

n_mels = 64

cmap = plt.get_cmap('gray')



e_file = '/kaggle/input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'#XC167210.mp3 XC124067.mp3



y, sr = librosa.load(e_file, mono=True, sr=sr)#duration=5

y = librosa.util.normalize(y, axis=0)

D = np.abs(librosa.stft(y, n_fft=int(0.040*sr),

                                hop_length=int(0.020*sr),

                                window='hann'))**2

S = librosa.feature.melspectrogram(S=D,sr=sr, n_mels=64)

plt.figure(figsize=(16,16))

sub = plt.subplot(3,1,1)

librosa.display.waveplot(y,sr=sr, x_axis='time')

sub.set_title('Wave')

sub2 = plt.subplot(3,1,2)

librosa.display.specshow(librosa.power_to_db(S,ref=np.max),

                                 y_axis='mel', fmax=8000,

                                 x_axis='time',

                                 cmap = cmap)

sub2.set_title('Log-melspectogram')

sub3 = plt.subplot(3,1,3)

rms = librosa.feature.rms(y=y,frame_length=1024, hop_length=512)

plt.semilogy(rms.T)

sub3.set_title('Energy')
sr=16000

n_mels = 64



def max_5s(file, sr):

    #open file

    y, sr = librosa.load(file, mono=True, sr=sr)

    y = librosa.util.normalize(y, axis=0)

    # if audio < 5s, append and cut

    if len(y) < 5*sr:

        for i in range(int(0.5+5*sr/len(y))):

            y = np.append(y,y)

        return y[:5*sr] #nothing more to do

    # get max energy point

    rms = librosa.feature.rms(y=y,frame_length=1024, hop_length=512)

    me = np.argmax(rms)*512

    # Check bounds (for audios >= 5s only)

    if me > 2.5*sr:

        if len(y) < me+2.5*sr: # check for upper bound

            y5 = y[int(me-2.5*sr):int(me+2.5*sr)] #2.5 seg before and 2.5 after

        else:

            y5 = y[len(y)-5*sr:len(y)] # get the last 5 s

    else:

        y5 = y[:5*sr] # get the first 5 s

    return y5



y = max_5s(e_file, sr)

librosa.display.waveplot(y,sr=sr, x_axis='time')
df = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

df.info()
# Selecting high-rated sound only

dff = df[df['rating'] > 4.0]

# Selecting shorter files only, less data to process

dff = dff[dff['duration'] < 20]

print(len(dff))
# Selecting birds with more than 10 examples left

dfc = dff.groupby('ebird_code')['ebird_code'].count()

dff = dff[~dff['ebird_code'].isin(dfc[dfc.values < 10].index)]

print(len(dff))
# Not all classes may be represented according to filtering

# Several classes decreased a lot

dfc = dff.groupby('ebird_code')['ebird_code'].count()

plt.figure(figsize=(16,8))

dfc.plot.bar()
audio_path = Path('/kaggle/input/birdsong-recognition/train_audio')



sound_categories = dff['ebird_code'].unique()



audios = []

Y = []

Y_classes = []

label = 0

for category_name in sound_categories:

    #Walk through the dataframe filename values

    l_files = dff[dff['ebird_code'] == category_name]['filename'].values

    for file_name in l_files:

        try:

            sound_path = audio_path/category_name/file_name

            y = max_5s(sound_path, sr)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mels)

            to_append = ''

            for e in mfcc:

                to_append += f' {np.mean(e)} {np.std(e)}'

            S = np.fromstring(to_append, dtype=float, sep=" ")

            audios.append(S)

            Y.append(label)

            Y_classes.append(category_name)

        except:

            print(sound_path)

            pass

    label +=1

    if label == 20:

        break

X = np.array(audios)

Y = np.array(Y)

num_classes = len(sound_categories)
perc_test = 0.2



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=perc_test)

print(x_train.shape)

print(x_test.shape)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Total de exemplos: "+ str(len(x_test)))



#score = clf.evaluate(x_test, y_test, verbose=0)

#print('Loss de Teste:', score[0])

print('Acurácia de Teste:', len(y_pred[y_pred==y_test])/len(y_pred))

#print(y_pred)

#print(y_test)



print('Confusion Matrix')

print(confusion_matrix(y_test, y_pred))

print('Classification Report')

print(classification_report(y_test, y_pred, target_names=np.unique(Y_classes)))
def load_test_clip(path, start_time, duration=5):

    return librosa.load(path, offset=start_time, duration=duration, sr=sr)[0]
TEST_FOLDER = '../input/birdsong-recognition/test_audio/'

test_info = pd.read_csv('../input/birdsong-recognition/test.csv')

test_info.head()
pred_limit = 0.6

def make_prediction(sound_clip, birds):

    mfcc = librosa.feature.mfcc(y=sound_clip, sr=sr, n_mfcc=n_mels)

    to_append = ''

    for e in mfcc:

        to_append += f' {np.mean(e)} {np.std(e)}'

    S = np.fromstring(to_append, dtype=float, sep=" ")

    ret = clf.predict_proba(S)

    pred = np.argmax(ret[0])

    if ret[0][pred] > pred_limit:

        return Y_classes(pred)

    else:

        return 'noclass'
try:

    preds = []

    for index, row in test_info.iterrows():

        # Get test row information

        site = row['site']

        start_time = row['seconds'] - 5

        row_id = row['row_id']

        audio_id = row['audio_id']



        # Get the test sound clip

        if site == 'site_1' or site == 'site_2':

            sound_clip = load_test_clip(TEST_FOLDER + audio_id + '.mp3', start_time)

        else:

            sound_clip = load_test_clip(TEST_FOLDER + audio_id + '.mp3', 0, duration=None)



        # Make the prediction

        pred = make_prediction(sound_clip, birds)



        # Store prediction

        preds.append([row_id, pred])



    preds = pd.DataFrame(preds, columns=['row_id', 'birds'])

except:

    preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
preds
preds.to_csv('submission.csv', index=False)