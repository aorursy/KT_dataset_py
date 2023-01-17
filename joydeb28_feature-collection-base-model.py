import numpy as np
import pandas as pd
import csv
import wave
from scipy.io import wavfile
import os
from sklearn.utils import shuffle
import sklearn
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, SimpleRNN

import librosa
import warnings
warnings.filterwarnings('ignore')
sequence_length = 50
class Preprocessing():
    def load_data(self,file_path):
        df = pd.read_csv(file_path,usecols = ['ebird_code','filename'])
        print(df.head())
        birds = df["ebird_code"].unique()
        self.id_to_bird = {k:v for k,v in enumerate(birds)}
        self.bird_to_id = {v:k for k,v in enumerate(birds)}
        df = shuffle(df)
        return df
    
    def get_features_(self,df,file_path):   
        to_append = f'bird chroma_stft rmse spec_cent spec_bw rolloff zcr mfcc'
        for i, item in df.iterrows():
            bird = self.bird_to_id[item['ebird_code']]
            audio_file = os.path.join(file_path,item['ebird_code'],item['filename'])
            print(audio_file)
            y, sr = librosa.load(audio_file)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
           
            to_append += f'{bird} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

    def get_feature(self,df,file_path):
        bird_list = list()
        audio_data = list()
        for i, item in df.iterrows():
            bird = self.bird_to_id[item['ebird_code']]
            audio_file = os.path.join(file_path,item['ebird_code'],item['filename'])
            
            wave_data, wave_rate = librosa.load(audio_file)
            data_point_per_second = 10
            prepared_sample = wave_data[0::int(wave_rate/data_point_per_second)]
            normalized_sample = sklearn.preprocessing.minmax_scale(prepared_sample, axis=0)

            song_sample = []
            sample_length = 5*data_point_per_second
            for idx in range(0,len(normalized_sample),sample_length): 
                song_sample = normalized_sample[idx:idx+sample_length]
                if len(song_sample)>=sample_length:
                    audio_data.append(np.asarray(song_sample).astype(np.float32))
                    bird_list.append(bird)
        data = pd.DataFrame({"audio_data":audio_data,"bird":bird})
        return data
    
    def get_data(self,df):
        train,valid = sklearn.model_selection.train_test_split(df,test_size=0.2, random_state=42)
        x_train = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in train["audio_data"]]),(train.shape[0],1,sequence_length))).astype(np.float32)
        groundtruth = np.asarray([np.asarray(x) for x in train["bird"]]).astype(np.float32)
        y_train = to_categorical(groundtruth, num_classes=len(self.bird_to_id.keys()), dtype='float32')

        x_valid = np.asarray(np.reshape(np.asarray([np.asarray(x) for x in valid["audio_data"]]),(valid.shape[0],1,sequence_length))).astype(np.float32)
        validation_groundtruth = np.asarray([np.asarray(x) for x in valid["bird"]]).astype(np.float32)
        y_valid = to_categorical(validation_groundtruth, num_classes=len(self.bird_to_id.keys()), dtype='float32')
        return x_train,y_train,x_valid,y_valid
prp_obj = Preprocessing()
train_df = prp_obj.load_data('/kaggle/input/birdsong-recognition/train.csv')
train_audio_path = '/kaggle/input/birdsong-recognition/train_audio/'
data = prp_obj.get_feature(train_df,train_audio_path)
data.head()
x_train,y_train,x_valid,y_valid = prp_obj.get_data(data)
bird2id = prp_obj.bird_to_id
id2bird = prp_obj.id_to_bird
num_class = len(bird2id.keys())
class CreateModel():
    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(32, return_sequences=True, recurrent_dropout=0.2,input_shape=(None, sequence_length)))
        self.model.add(LSTM(32,recurrent_dropout=0.2))
        self.model.add(Dense(128,activation = 'relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(128,activation = 'relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(num_class, activation="softmax"))
        self.model.summary()
        self.model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['acc'])
        
    def run(self):
        callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),
                     EarlyStopping(monitor='val_loss', patience=10),
                     ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)]
        self.history = self.model.fit(x_train, y_train, 
                                      epochs = 100, 
                                      batch_size = 32,
                                      validation_data=(x_valid, y_valid), 
                                      callbacks=callbacks)
model_obj = CreateModel()
model_obj.build_model()
model_obj.run()
class Prediction():
    def __init__(self):
        self.model = keras.models.load_model("model.h5")
        test_file_path = "/kaggle/input/birdsong-recognition/example_test_audio"
        test_df = pd.read_csv("/kaggle/input/birdsong-recognition/example_test_audio_summary.csv")
        test_df["audio_id"] = [ "BLKFR-10-CPL_20190611_093000.pt540" if filename=="BLKFR-10-CPL" else "ORANGE-7-CAP_20190606_093000.pt623" for filename in test_df["filename"]]

    def predict_submission(self, df, audio_file_path):
        
        loaded_audio_sample = []
        previous_filename = ""
        data_point_per_second = 10
        sample_length = 5*data_point_per_second
        wave_data = []
        wave_rate = None

        for idx,row in df.iterrows():
            try:
                if previous_filename == "" or previous_filename!=row.audio_id:
                    filename = '{}/{}.mp3'.format(audio_file_path, row.audio_id)
                    wave_data, wave_rate = librosa.load(filename)
                    prepared_sample = wave_data[0::int(wave_rate/data_point_per_second)]
                    sample = sklearn.preprocessing.minmax_scale(prepared_sample, axis=0)
                previous_filename = row.audio_id

                #basically allows to check if we are running the examples or the test set.
                if "site" in df.columns:
                    if row.site=="site_1" or row.site=="site_2":
                        song_sample = np.array(sample[int(row.seconds-5)*data_point_per_second:int(row.seconds)*data_point_per_second])
                    elif row.site=="site_3":
                        song_sample = np.array(sample[0:sample_length])
                else:
                    song_sample = np.array(sample[int(row.seconds-5)*data_point_per_second:int(row.seconds)*data_point_per_second])

                input_data = np.reshape(np.asarray([song_sample]),(1,sequence_length)).astype(np.float32)
                prediction = model.predict(np.array([input_data]))

                if any(prediction[0]>0.5):
                    predicted_bird = id2bird[np.argmax(prediction)]
                    df.at[idx,"birds"] = predicted_bird
                else:
                    df.at[idx,"birds"] = "nocall"
            except:
                df.at[idx,"birds"] = "nocall"
        return df
    
    def make_submission(self):
        test_file_path = "/kaggle/input/birdsong-recognition/test_audio"
        test_df = pd.read_csv("/kaggle/input/birdsong-recognition/test.csv")
        submission_df = pd.read_csv("/kaggle/input/birdsong-recognition/sample_submission.csv")
        submission_df = self.predict_submission(test_df, test_file_path)

        submission_df[["row_id","birds"]].to_csv('submission.csv', index=False)
        submission_df.head()
pred_obj = Prediction()
pred_obj.make_submission()