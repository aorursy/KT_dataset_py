!cd /kaggle/input/audio2numpy

!cp -r /kaggle/input/audio2numpy/* ./

from audio2numpy import open_audio
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



import keras

import tensorflow.keras.layers as L

from tensorflow.keras.models import Model

import tensorflow as tf 



import glob as glob

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score



from joblib import Parallel, delayed

from tqdm import tqdm_notebook



from scipy.sparse import csr_matrix



from sklearn.preprocessing import LabelEncoder

from transformers import BertConfig,TFBertModel,BertModel



import gc

import librosa
NUM_BINS = 512

MAX_LEN = 512

def audio_read(path):

    recording, sr = open_audio(path)

    

    if recording.shape[0] != recording.size:

        return recording.mean(axis=1) 

    else:

        return recording

############################################################

def tokenize(path, NUM_BINS = NUM_BINS,MAX_LEN=MAX_LEN):

    signal = np.resize(audio_read(path), (MAX_LEN,))

    signal_bins = np.linspace(signal.min(), signal.max(), NUM_BINS + 1)

    signal = np.digitize(signal, bins=signal_bins) - 1 

    signal = np.minimum(signal, NUM_BINS - 1)

    return signal.astype(int)
def tokenize_list(lis, NUM_BINS = NUM_BINS,MAX_LEN=MAX_LEN):

    signals = np.array([Parallel(n_jobs=4)(delayed(tokenize)(filename) for filename in tqdm_notebook(train_audio[:max_len]))])[0]   

    return signals
def build_model(MAX_LEN = MAX_LEN, NUM_BINS = NUM_BINS):

    ids = L.Input((MAX_LEN,), dtype=tf.int32)

    config = BertConfig() 

    config.vocab_size = NUM_BINS

    config.num_hidden_layers = 3

    bert_model = TFBertModel(config=config)



    x = bert_model(ids)[0]

    x = L.Flatten()(x)

    x = L.Dense(264,activation='softmax')(x)

    

    model = Model(inputs=ids, outputs=x)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = 'accuracy')



    return model
model = build_model()
model.summary()
train_audio = []

for directory in sorted(glob.glob('/kaggle/input/birdsong-recognition/train_audio/*')):

    train_audio.extend(sorted(glob.glob(directory+'/*')))

train_audio = np.array(train_audio)
max_len = len(train_audio)
%%time

#1h 19m for all 21_000

X = np.array([Parallel(n_jobs=4)(delayed(tokenize)(filename) for filename in tqdm_notebook(train_audio[:max_len]))])[0]
## Preprocess dataset and create validation sets

le = LabelEncoder()

seed = 43

train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

le.fit(train['ebird_code'])

Y = pd.get_dummies(train['ebird_code'])[:X.shape[0]].values.reshape((-1,264)).astype(int)

y = pd.Series(le.fit_transform(train['ebird_code'])[:X.shape[0]].ravel())
y.shape,Y.shape
X.shape
with tf.device('/gpu:0'):

    model.fit(X,Y,epochs=15)
val_preds = model.predict(X)
accuracy_score(np.argmax(Y,axis=1),np.argmax(val_preds,axis=1))
test = pd.read_csv('/kaggle/input/birdsong-recognition/test.csv')

TEST_FOLDER = '../input/birdsong-recognition/test_audio/'

try:

    print('predicting')

    preds = []

    for index, row in test.iterrows():

        # Get test row information

        site = row['site']

        start_time = row['seconds'] - 5

        row_id = row['row_id']

        audio_id = row['audio_id']



        # Get the test sound clip

        if site == 'site_1' or site == 'site_2':

            x = tokenize(TEST_FOLDER + audio_id + '.mp3')

        else:

            x = tokenize(TEST_FOLDER + audio_id + '.mp3')

        

#         x = extract_features(TEST_FOLDER + audio_id + '.mp3').reshape(1, -1)

        # Make the prediction

        pred = le.inverse_transform(np.argmax(model.predict(x).flatten(),axis=1))

#         pred = le.inverse_transform(clf.predict(nan_remove(x.flatten().reshape(1, -1))))

#         print(pred)

        # Store prediction

        preds.append([row_id, pred])

except:

     preds = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')

        

preds = pd.DataFrame(preds, columns=['row_id', 'birds'])

preds.to_csv('submission.csv', index = False)