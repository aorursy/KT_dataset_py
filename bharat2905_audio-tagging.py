# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_csv_file=pd.read_csv('../input/train.csv')

train_csv_file
print(os.listdir("../input/audio_train/audio_train"))
filename_label={"../input/audio_train/audio_train/"+k:v for k,v in zip(train_csv_file.fname.values, train_csv_file.label.values)}
filename_label['../input/audio_train/audio_train/fff81f55.wav']
import librosa


def audio_norm(data):



    max_data = np.max(data)

    min_data = np.min(data)

    data = (data-min_data)/(max_data-min_data)

    return data



input_length = 16000*2

def load_audio_file(file_path, input_length=input_length):

    data = librosa.core.load(file_path, sr=16000)[0] #, sr=16000

    if len(data)>input_length:

        

        

        max_offset = len(data)-input_length

        

        offset = np.random.randint(max_offset)

        

        data = data[offset:(input_length+offset)]

        

        

    else:

        

        if input_length > len(data):

            max_offset = input_length - len(data)



            offset = np.random.randint(max_offset)

        else:

            offset = 0

        

        

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        

        

    data = audio_norm(data)

    return data
import glob

train_files = glob.glob("../input/audio_train/audio_train/*.wav")
train_files
import matplotlib.pyplot as plt
data_base = load_audio_file(train_files[9])

fig = plt.figure(figsize=(14, 8))

plt.title('Raw wave : %s ' % (filename_label[train_files[0]]))

plt.ylabel('Amplitude')

plt.plot(np.linspace(0, 1, input_length), data_base)

plt.show()
list_labels = sorted(list(set(train_csv_file.label.values)))
list_labels
label_to_int = {k:v for v,k in enumerate(list_labels)}
label_to_int
int_to_label = {v:k for k,v in label_to_int.items()}
int_to_label
file_to_int = {k:label_to_int[v] for k,v in filename_label.items()}
file_to_int
n_class=41
from keras.models import Sequential

from keras.layers import Conv1D,MaxPool1D,Dropout,GlobalMaxPool1D,Dense
model=Sequential()

model.add(Conv1D(16,kernel_size=9,activation='relu',input_shape=(input_length,1)))

model.add(Conv1D(16,kernel_size=9,activation='relu'))

model.add(MaxPool1D(pool_size=16))

model.add(Dropout(rate=0.1))

model.add(Conv1D(32,kernel_size=3,activation='relu'))

model.add(Conv1D(32,kernel_size=3,activation='relu'))

model.add(MaxPool1D(pool_size=4))

model.add(Dropout(rate=0.1))

model.add(Conv1D(32,kernel_size=3,activation='relu'))

model.add(Conv1D(32,kernel_size=3,activation='relu'))

model.add(MaxPool1D(pool_size=4))

model.add(Dropout(rate=0.1))

model.add(Conv1D(64,kernel_size=3,activation='relu'))

model.add(Conv1D(64,kernel_size=3,activation='relu'))

model.add(GlobalMaxPool1D())

model.add(Dropout(rate=0.2))

model.add(Dense(128,activation='relu'))

model.add(Dense(256,activation='relu'))

model.add(Dense(n_class,activation='softmax'))

          

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()
batch_size=32
from random import shuffle

from sklearn.model_selection import train_test_split
def chunker(seq, size):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
def train_generator(list_files, batch_size=batch_size):

    while True:

        shuffle(list_files)

        for batch_files in chunker(list_files, size=batch_size):

            batch_data = [load_audio_file(fpath) for fpath in batch_files]

            batch_data = np.array(batch_data)[:,:,np.newaxis]

            batch_labels = [file_to_int[fpath] for fpath in batch_files]

            batch_labels = np.array(batch_labels)

            

            yield batch_data, batch_labels
tr_files, val_files = train_test_split(train_files, test_size=0.1)
len(tr_files)
model.fit_generator(train_generator(tr_files), steps_per_epoch=len(tr_files)//batch_size, epochs=2,

                    validation_data=train_generator(val_files), validation_steps=len(val_files)//batch_size,use_multiprocessing=True, workers=8, max_queue_size=20)
model.save_weights("audio_tagging.h5")