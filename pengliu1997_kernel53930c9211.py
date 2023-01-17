# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
DATA_PATH = '../input/infantcry/train/'
DATA_TEST_PATH = '../input/infantcry/test'








labels, _, _ = get_labels(DATA_PATH)

for label in labels:
    mfcc_vectors = []

    wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
    print(wavfiles)
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
        mfcc = np.zeros((20, 400))
        mfcc_feat = wav2mfcc(wavfile)[:, :400]
        mfcc[:, :mfcc_feat.shape[1]] = mfcc_feat
        mfcc_vectors.append(mfcc)

    mfcc_vectors = np.stack(mfcc_vectors)


    np.save(label + '.npy', mfcc_vectors)

        

def save_data_to_array_test(path=DATA_TEST_PATH):
    mfcc_vectors = []
        
    wavfiles = [DATA_TEST_PATH + '/' + wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format('test')):
        mfcc = np.zeros((20, 400))
        mfcc_feat = wav2mfcc(wavfile)[:, :400]
        mfcc[:, :mfcc_feat.shape[1]] = mfcc_feat
        mfcc_vectors.append(mfcc)
            
    mfcc_vectors = np.stack(mfcc_vectors)
    np.save('test.npy', mfcc_vectors)
        


def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 400, channel)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


feature_dim_2 = 32

save_data_to_array() 
save_data_to_array_test()

X, Y = get_train_test()
skf = StratifiedKFold(n_splits=5)
ls
for idx, (tr_idx, val_idx) in enumerate(skf.split(X, Y)):
    print(idx)

    feature_dim_1 = 20
    channel = 1
    epochs = 50
    batch_size = 32
    verbose = 1
    num_classes = 6

    X_train, X_test = X[tr_idx], X[val_idx]
    y_train, y_test = Y[tr_idx], Y[val_idx]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channel) / 255.0
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], channel) / 255.0

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    
    model = get_model()

    my_callbacks = [
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint(filepath='model-{0}.h5'.format(idx), save_best_only=True),
    ]

    model.fit(X_train, y_train_hot, 
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=verbose, 
              validation_data=(X_test, y_test_hot),
              callbacks=my_callbacks
             )
    model.load_weights('model-{0}.h5'.format(idx))
    

test_pred = np.zeros((228, 6))
for path in ['model-0.h5', 'model-2.h5', 'model-3.h5'][:1]:
    model.load_weights(path)
    
    X_test = np.load('test.npy') / 255.0
    test_pred += model.predict(X_test.reshape(228, 20, 400, 1))



wavfiles = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]   

import pandas as pd
df = pd.DataFrame()

df['id'] = [wavfile for wavfile in os.listdir(DATA_TEST_PATH)]
df['label'] = [['awake','diaper','hug', 'hungry','sleepy', 'uncomfortable'][x] for x in test_pred.argmax(1)]
df.to_csv('baseline.csv', index=None)