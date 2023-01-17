# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings                        # To ignore any warnings

warnings.filterwarnings("ignore")

%matplotlib inline

%pylab inline

import os

import pandas as pd

import librosa

import librosa.display

import IPython.display as ipd

import glob 

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

import wave

from scipy.io import wavfile

import os

import librosa

import warnings

from sklearn.utils import shuffle

import sklearn

from tqdm import tqdm



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers

from tensorflow.keras import Input

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, SimpleRNN



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
train_df = pd.read_csv('../input/birdsong-recognition/train.csv')

train_df.head()
test = pd.read_csv('../input/birdsong-recognition/test.csv',)

test.head()
# Y variable - Ebird Code 



print('Number of Unique Birds in the the Dataset is: ' + str(train_df['ebird_code'].nunique()))



# Distribution of the labels



train_df['ebird_code'].value_counts().plot.bar()
# Play the firt clip for an aldfly

aldfly = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'

y,sr = librosa.load(aldfly, sr=None)

ipd.Audio(aldfly) 

plt.figure(figsize=(14, 5))

librosa.display.waveplot(y,sr = sr)
Y = librosa.stft(y)

Xdb = librosa.amplitude_to_db(abs(Y))

plt.figure(figsize=(14, 5))

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

plt.colorbar()
#Using Francois's code to extract the data/ run model





def get_sample(filename, bird, samples_df):

    wave_data, wave_rate = librosa.load(filename)

    data_point_per_second = 10

    

    #Take 10 data points every second

    prepared_sample = wave_data[0::int(wave_rate/data_point_per_second)]

    #We normalize each sample before extracting 5s samples from it

    normalized_sample = sklearn.preprocessing.minmax_scale(prepared_sample, axis=0)

    

    #only take 5s samples and add them to the dataframe

    song_sample = []

    sample_length = 5*data_point_per_second

    for idx in range(0,len(normalized_sample),sample_length): 

        song_sample = normalized_sample[idx:idx+sample_length]

        if len(song_sample)>=sample_length:

            samples_df = samples_df.append({"song_sample":np.asarray(song_sample).astype(np.float32),

                                            "bird":ebird_to_id[bird]}, 

                                           ignore_index=True)

    return samples_df





birds_selected = shuffle(train_df["ebird_code"].unique())

train_df = train_df.query("ebird_code in @birds_selected")



ebird_to_id = {}

id_to_ebird = {}

ebird_to_id["nocall"] = 0

id_to_ebird[0] = "nocall"

for idx, unique_ebird_code in enumerate(train_df.ebird_code.unique()):

    ebird_to_id[unique_ebird_code] = str(idx+1)

    id_to_ebird[idx+1] = str(unique_ebird_code)
%%time

warnings.filterwarnings("ignore")

samples_df = pd.DataFrame(columns=["song_sample","bird"])



#We limit the number of audio files being sampled to 5000 in this notebook to save time

#However, we have already limited the number of bird species

sample_limit = 5000

with tqdm(total=sample_limit) as pbar:

    for idx, row in train_df[:sample_limit].iterrows():

        pbar.update(1)

        audio_file_path = "/kaggle/input/birdsong-recognition/train_audio/"

        audio_file_path += row.ebird_code

        samples_df = get_sample('{}/{}'.format(audio_file_path, row.filename), row.ebird_code, samples_df)
samples_df = shuffle(samples_df)

samples_df[:10]
sequence_length = 50

training_percentage = 0.9

training_item_count = int(len(samples_df)*training_percentage)

validation_item_count = len(samples_df)-int(len(samples_df)*training_percentage)

training_df = samples_df[:training_item_count]

validation_df = samples_df[training_item_count:]
#Base Model, lots of room for improvement





model = Sequential()

model.add(LSTM(32, return_sequences=True, recurrent_dropout=0.2,input_shape=(None, sequence_length)))

model.add(LSTM(32,recurrent_dropout=0.2))

model.add(Dense(128,activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(128,activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(len(ebird_to_id.keys()), activation="softmax"))



model.summary()



callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),

             EarlyStopping(monitor='val_loss', patience=10),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model.compile(loss="categorical_crossentropy", optimizer='adam')
X_train = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in training_df["song_sample"]]),(training_item_count,1,sequence_length))).astype(np.float32)

groundtruth = np.asarray([np.asarray(x) for x in training_df["bird"]]).astype(np.float32)

Y_train = to_categorical(

                groundtruth, num_classes=len(ebird_to_id.keys()), dtype='float32'

            )





X_validation = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in validation_df["song_sample"]]),(validation_item_count,1,sequence_length))).astype(np.float32)

validation_groundtruth = np.asarray([np.asarray(x) for x in validation_df["bird"]]).astype(np.float32)

Y_validation = to_categorical(

                validation_groundtruth, num_classes=len(ebird_to_id.keys()), dtype='float32'

            )
history = model.fit(X_train, Y_train, 

          epochs = 100, 

          batch_size = 32, 

          validation_data=(X_validation, Y_validation), 

          callbacks=callbacks)



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss over epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
model.load_weights("best_model.h5")



def predict_submission(df, audio_file_path):

        

    loaded_audio_sample = []

    previous_filename = ""

    data_point_per_second = 10

    sample_length = 5*data_point_per_second

    wave_data = []

    wave_rate = None

    

    for idx,row in df.iterrows():

        if previous_filename == "" or previous_filename!=row.filename:

            filename = '{}/{}.mp3'.format(audio_file_path, row.filename)

            wave_data, wave_rate = librosa.load(filename)

            sample = wave_data[0::int(wave_rate/data_point_per_second)]

        previous_filename = row.filename

        

        #basically allows to check if we are running the examples or the test set.

        if "site" in df.columns:

            if row.site=="site_1" or row.site=="site_2":

                song_sample = np.array(sample[int(row.seconds-5)*data_point_per_second:int(row.seconds)*data_point_per_second])

            elif row.site=="site_3":

                #for now, I only take the first 5s of the samples from site_3 as they are groundtruthed at file level

                song_sample = np.array(sample[0:sample_length])

        else:

            #same as the first condition but I isolated it for later and it is for the example file

            song_sample = np.array(sample[int(row.seconds-5)*data_point_per_second:int(row.seconds)*data_point_per_second])



        input_data = np.reshape(np.asarray([song_sample]),(1,sequence_length)).astype(np.float32)

        prediction = model.predict(np.array([input_data]))

        predicted_bird = id_to_ebird[np.argmax(prediction)]



        df.at[idx,"birds"] = predicted_bird

    return df
audio_file_path = "/kaggle/input/birdsong-recognition/example_test_audio"

example_df = pd.read_csv("/kaggle/input/birdsong-recognition/example_test_audio_summary.csv")

example_df["filename"] = [ "BLKFR-10-CPL_20190611_093000.pt540" if filename=="BLKFR-10-CPL" else "ORANGE-7-CAP_20190606_093000.pt623" for filename in example_df["filename"]]





if os.path.exists(audio_file_path):

    example_df = predict_submission(example_df, audio_file_path)

example_df
audio_file_path = "/kaggle/input/birdsong-recognition/test_audio/"

test_df = pd.read_csv("/kaggle/input/birdsong-recognition/test.csv")

submission_df = pd.read_csv("/kaggle/input/birdsong-recognition/sample_submission.csv")



if os.path.exists(audio_file_path):

    submission_df = predict_submission(test_df, audio_file_path)
submission_df[["row_id","birds"]].to_csv('submission.csv', index=False)

submission_df.head()