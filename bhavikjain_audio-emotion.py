# Import libraries 

import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import glob 

from sklearn.metrics import confusion_matrix

import IPython.display as ipd  # To play sound in the notebook

import os

import sys

import warnings

# ignore warnings 

if not sys.warnoptions:

    warnings.simplefilter("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 
#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"

RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

SAVEE = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"

CREMA = "/kaggle/input/cremad/AudioWAV/"



# Run one example 

dir_list = os.listdir(SAVEE)

dir_list[0:5]
# Get the data location for SAVEE

dir_list = os.listdir(SAVEE)



# parse the filename to get the emotions

emotion=[]

path = []

for i in dir_list:

    if i[-8:-6]=='_a':

        emotion.append('male_angry')

    elif i[-8:-6]=='_d':

        emotion.append('male_disgust')

    elif i[-8:-6]=='_f':

        emotion.append('male_fear')

    elif i[-8:-6]=='_h':

        emotion.append('male_happy')

    elif i[-8:-6]=='_n':

        emotion.append('male_neutral')

    elif i[-8:-6]=='sa':

        emotion.append('male_sad')

    elif i[-8:-6]=='su':

        emotion.append('male_surprise')

    else:

        emotion.append('male_error') 

    path.append(SAVEE + i)

    

# Now check out the label count distribution 

SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])

SAVEE_df['source'] = 'SAVEE'

SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)

SAVEE_df.labels.value_counts()
dir_list = os.listdir(RAV)

dir_list.sort()



emotion = []

gender = []

path = []

for i in dir_list:

    fname = os.listdir(RAV + i)

    for f in fname:

        part = f.split('.')[0].split('-')

        emotion.append(int(part[2]))

        temp = int(part[6])

        if temp%2 == 0:

            temp = "female"

        else:

            temp = "male"

        gender.append(temp)

        path.append(RAV + i + '/' + f)



        

RAV_df = pd.DataFrame(emotion)

RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})

RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)

RAV_df.columns = ['gender','emotion']

RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion

RAV_df['source'] = 'RAVDESS'  

RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)

RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)

RAV_df.labels.value_counts()
dir_list = os.listdir(TESS)

dir_list.sort()

dir_list
path = []

emotion = []



for i in dir_list:

    fname = os.listdir(TESS + i)

    for f in fname:

        if i == 'OAF_angry' or i == 'YAF_angry':

            emotion.append('female_angry')

        elif i == 'OAF_disgust' or i == 'YAF_disgust':

            emotion.append('female_disgust')

        elif i == 'OAF_Fear' or i == 'YAF_fear':

            emotion.append('female_fear')

        elif i == 'OAF_happy' or i == 'YAF_happy':

            emotion.append('female_happy')

        elif i == 'OAF_neutral' or i == 'YAF_neutral':

            emotion.append('female_neutral')                                

        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':

            emotion.append('female_surprise')               

        elif i == 'OAF_Sad' or i == 'YAF_sad':

            emotion.append('female_sad')

        else:

            emotion.append('Unknown')

        path.append(TESS + i + "/" + f)



TESS_df = pd.DataFrame(emotion, columns = ['labels'])

TESS_df['source'] = 'TESS'

TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)

TESS_df.labels.value_counts()
dir_list = os.listdir(CREMA)

dir_list.sort()

print(dir_list[0:10])
gender = []

emotion = []

path = []

female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,

          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]



for i in dir_list: 

    part = i.split('_')

    if int(part[0]) in female:

        temp = 'female'

    else:

        temp = 'male'

    gender.append(temp)

    if part[2] == 'SAD' and temp == 'male':

        emotion.append('male_sad')

    elif part[2] == 'ANG' and temp == 'male':

        emotion.append('male_angry')

    elif part[2] == 'DIS' and temp == 'male':

        emotion.append('male_disgust')

    elif part[2] == 'FEA' and temp == 'male':

        emotion.append('male_fear')

    elif part[2] == 'HAP' and temp == 'male':

        emotion.append('male_happy')

    elif part[2] == 'NEU' and temp == 'male':

        emotion.append('male_neutral')

    elif part[2] == 'SAD' and temp == 'female':

        emotion.append('female_sad')

    elif part[2] == 'ANG' and temp == 'female':

        emotion.append('female_angry')

    elif part[2] == 'DIS' and temp == 'female':

        emotion.append('female_disgust')

    elif part[2] == 'FEA' and temp == 'female':

        emotion.append('female_fear')

    elif part[2] == 'HAP' and temp == 'female':

        emotion.append('female_happy')

    elif part[2] == 'NEU' and temp == 'female':

        emotion.append('female_neutral')

    else:

        emotion.append('Unknown')

    path.append(CREMA + i)

    

CREMA_df = pd.DataFrame(emotion, columns = ['labels'])

CREMA_df['source'] = 'CREMA'

CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)

CREMA_df.labels.value_counts()
df = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis = 0)

print(df.labels.value_counts())

df.head()

df.to_csv("Data_path.csv",index=False)
ref = pd.read_csv("Data_path.csv")

ref.head()
df = pd.DataFrame(columns=['feature'])



# loop feature extraction over the entire dataset

counter=0

for index,path in enumerate(ref.path):

    X, sample_rate = librosa.load(path

                                  , res_type='kaiser_fast'

                                  ,duration=3

                                  ,sr=44100

                                  ,offset=0.5

                                 )

    sample_rate = np.array(sample_rate)

    

    # mean as the feature. Could do min and max etc as well. 

    mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                        sr=sample_rate, 

                                        n_mfcc=13),

                    axis=0)

    df.loc[counter] = [mfccs]

    counter=counter+1   



# Check a few records to make sure its processed successfully

print(len(df))

df.head()
df = pd.concat([ref,pd.DataFrame(df['feature'].values.tolist())],axis=1)

df.head()
df.isnull().sum()
df=df.fillna(0)

print(df.shape)
import keras

from keras import regularizers

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model, model_from_json

from keras.layers import Dense, Embedding, LSTM

from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.utils import np_utils, to_categorical

from keras.callbacks import ModelCheckpoint



# sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Other  

import librosa

import librosa.display

import json

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import seaborn as sns

import glob 

import os

import pickle

import IPython.display as ipd  
def noise(data):

    """

    Adding White Noise.

    """

    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html

    noise_amp = 0.05*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5

    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])

    return data

    

def shift(data):

    """

    Random Shifting.

    """

    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500

    return np.roll(data, s_range)

    

def stretch(data, rate=0.8):

    """

    Streching the Sound. Note that this expands the dataset slightly

    """

    data = librosa.effects.time_stretch(data, rate)

    return data

    

def pitch(data, sample_rate):

    """

    Pitch Tuning.

    """

    bins_per_octave = 12

    pitch_pm = 2

    pitch_change =  pitch_pm * 2*(np.random.uniform())   

    data = librosa.effects.pitch_shift(data.astype('float64'), 

                                      sample_rate, n_steps=pitch_change, 

                                      bins_per_octave=bins_per_octave)

    return data

    

def dyn_change(data):

    """

    Random Value Change.

    """

    dyn_change = np.random.uniform(low=-0.5 ,high=7)  # default low = 1.5, high = 3

    return (data * dyn_change)

    

def speedNpitch(data):

    """

    peed and Pitch Tuning.

    """

    # you can change low and high here

    length_change = np.random.uniform(low=0.8, high = 1)

    speed_fac = 1.2  / length_change # try changing 1.0 to 2.0 ... =D

    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)

    minlen = min(data.shape[0], tmp.shape[0])

    data *= 0

    data[0:minlen] = tmp[0:minlen]

    return data

ref = pd.read_csv("Data_path.csv")

ref.head()
from tqdm import tqdm
df = pd.DataFrame(columns=['feature'])

df_noise = pd.DataFrame(columns=['feature'])

df_speedpitch = pd.DataFrame(columns=['feature'])

cnt = 0



# loop feature extraction over the entire dataset

for i in tqdm(ref.path):

    

    # first load the audio 

    X, sample_rate = librosa.load(i

                                  , res_type='kaiser_fast'

                                  ,duration=3

                                  ,sr=44100

                                  ,offset=0.5

                                 )



    # take mfcc and mean as the feature. Could do min and max etc as well. 

    mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                        sr=np.array(sample_rate), 

                                        n_mfcc=13),

                    axis=0)

    

    df.loc[cnt] = [mfccs]   



    # random shifting (omit for now)

    # Stretch

    # pitch (omit for now)

    # dyn change

    

    # noise 

    aug = noise(X)

    aug = np.mean(librosa.feature.mfcc(y=aug, 

                                    sr=np.array(sample_rate), 

                                    n_mfcc=13),    

                  axis=0)

    df_noise.loc[cnt] = [aug]



    # speed pitch

    aug = speedNpitch(X)

    aug = np.mean(librosa.feature.mfcc(y=aug, 

                                    sr=np.array(sample_rate), 

                                    n_mfcc=13),    

                  axis=0)

    df_speedpitch.loc[cnt] = [aug]   



    cnt += 1



df.head()
df = pd.concat([ref,pd.DataFrame(df['feature'].values.tolist())],axis=1)

df_noise = pd.concat([ref,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)

df_speedpitch = pd.concat([ref,pd.DataFrame(df_speedpitch['feature'].values.tolist())],axis=1)

print(df.shape,df_noise.shape,df_speedpitch.shape)
df = pd.concat([df,df_noise,df_speedpitch],axis=0,sort=False)

df=df.fillna(0)

del df_noise, df_speedpitch



df.head()
def r(x):

    if x=='male_surprise':

        return 'surprise'

    if x=='male_disgust':

        return 'disgust'

    if x=='male_neutral':

        return 'neutral'

    if x=='male_angry':

        return 'angry'

    if x=='male_fear':

        return 'fear'

    if x=='male_sad':

        return 'sad'

    if x=='male_happy':

        return 'happy'

    if x=='female_disgust':

        return 'disgust'

    if x=='female_surprise':

        return 'surprise'

    if x=='female_happy':

        return 'happy'

    if x=='female_angry':

        return 'angry'

    if x=='female_neutral':

        return 'neutral'

    if x=='female_fear':

        return 'fear'

    if x=='female_sad':

        return 'sad'
y = df.labels.apply(r)
y.unique()
X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','labels','source'],axis=1)

                                                    , y

                                                    , test_size=0.20

                                                    , shuffle=True

                                                    , random_state=42

                                                   )
mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)



X_train = (X_train - mean)/std

X_test = (X_test - mean)/std
X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



# one hot encode the target 

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))

y_test = np_utils.to_categorical(lb.fit_transform(y_test))



print(X_train.shape)

print(lb.classes_)

#print(y_train[0:10])

#print(y_test[0:10])



# Pickel the lb object for future use 

filename = 'labels'

outfile = open(filename,'wb')

pickle.dump(lb,outfile)

outfile.close()
X_train = np.expand_dims(X_train, axis=2)

X_test = np.expand_dims(X_test, axis=2)

X_train.shape
y_train.shape
model = Sequential()

model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns

model.add(Activation('relu'))

model.add(Conv1D(256, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(7)) # Target class number

model.add(Activation('softmax'))

# opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

# opt = keras.optimizers.Adam(lr=0.0001)

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

model_history=model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))
model_name = 'Emotion_Model.h5'

save_dir = os.path.join(os.getcwd(), 'saved_models')



if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Save model and weights at %s ' % model_path)



# Save the model to disk

model_json = model.to_json()

with open("model_json.json", "w") as json_file:

    json_file.write(model_json)
json_file = open('model_json.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



# load weights into new model

loaded_model.load_weights("saved_models/Emotion_Model.h5")

print("Loaded model from disk")

 

# Keras optimiser

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

score = loaded_model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
preds = loaded_model.predict(X_test, 

                         batch_size=16, 

                         verbose=1)



preds=preds.argmax(axis=1)

preds
preds = preds.astype(int).flatten()

preds = (lb.inverse_transform((preds)))

preds = pd.DataFrame({'predictedvalues': preds})



# Actual labels

actual=y_test.argmax(axis=1)

actual = actual.astype(int).flatten()

actual = (lb.inverse_transform((actual)))

actual = pd.DataFrame({'actualvalues': actual})



# Lets combined both of them into a single dataframe

finaldf = actual.join(preds)
classes = finaldf.actualvalues.unique()

classes.sort()    

print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes))
from keras.models import Sequential, Model, model_from_json

import matplotlib.pyplot as plt

import keras 

import pickle

import wave  # !pip install wave

import os

import pandas as pd

import numpy as np

import sys

import warnings

import librosa

import librosa.display

import IPython.display as ipd  # To play sound in the notebook



# ignore warnings 

if not sys.warnoptions:

    warnings.simplefilter("ignore")
CHUNK =1024

FORMAT = pyaudio.paInt16 

CHANNELS = 2 

RATE = 44100 

RECORD_SECONDS = 4

WAVE_OUTPUT_FILENAME = "test audio\\testing.wav"



p = pyaudio.PyAudio()



stream = p.open(format=FORMAT,

                channels=CHANNELS,

                rate=RATE,

                input=True,

                frames_per_buffer=CHUNK) #buffer



print("* recording")



frames = []



for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

    data = stream.read(CHUNK)

    frames.append(data) # 2 bytes(16 bits) per channel



print("* done recording")



stream.stop_stream()

stream.close()

p.terminate()



wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

wf.setnchannels(CHANNELS)

wf.setsampwidth(p.get_sample_size(FORMAT))

wf.setframerate(RATE)

wf.writeframes(b''.join(frames))

wf.close()

json_file = open('model_json.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)



# load weights into new model

loaded_model.load_weights("saved_models/Emotion_Model.h5")

print("Loaded model from disk")



# the optimiser

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
data, sampling_rate = librosa.load('/kaggle/input/abcdefghi/akshat.wav')

ipd.Audio('/kaggle/input/abcdefghi/akshat.wav')
X, sample_rate = librosa.load('/kaggle/input/abcdefghi/akshat.wav'

                              ,res_type='kaiser_fast'



                              ,sr=44100

                              ,offset=0.5

                             )



sample_rate = np.array(sample_rate)

mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)

newdf = pd.DataFrame(data=mfccs).T

newdf
newdf= np.expand_dims(newdf, axis=2)

newpred = loaded_model.predict(newdf, 

                         batch_size=16, 

                         verbose=1)



newpred
final = newpred.argmax(axis=1)

final = final.astype(int).flatten()

final = (lb.inverse_transform((final)))

print(final) #emo(final) #gender(final)