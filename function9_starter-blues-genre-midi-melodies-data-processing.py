!pip install mido
import mido # easy to use python MIDI library

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure



from mido import MidiFile, MidiTrack, Message
os.listdir('../input')
paths = []

songs = []

#append every filepath in the blues folder to paths[]

for r, d, f in os.walk(r'../input/blues'):

    for file in f:

        if '.mid' in file:

            paths.append(os.path.join(r, file))



#for each path in the array, create a Mido object and append it to song[]

for path in paths:

    mid = MidiFile(path, type = 1)

    songs.append(mid)

del paths
#first Mido Object

print(songs[0])
notes = []

dataset = []

chunk = []



#for each in midi object in list of songs

for i in range(len(songs)):

    #for each note in midi object

    for msg in songs[i]:

        #filtering out meta messages

        if not msg.is_meta:

            #filtering out control changes

            if (msg.type == 'note_on'):

                #normalizing note and velocity values

                notes.append(msg.note)

    for i in range(1, len(notes)):

        chunk.append(notes[i])

    dataset.append(chunk)

    chunk = []

    notes = []

del chunk

del notes
print(dataset[0])
dataset = np.array(dataset)

dataset.shape
notes = []

dataset = []

chunk = []



#for each in midi object in list of songs

for i in range(len(songs)):

    #for each note in midi object

    for msg in songs[i]:

        #filtering out meta messages

        if not msg.is_meta:

            #filtering out control changes

            if (msg.type == 'note_on'):

                #normalizing note and velocity values

                notes.append(msg.note)

    for i in range(1, len(notes)):

        chunk.append(notes[i])

        #save each 16 note chunk

        if (i % 16 == 0):

            dataset.append(chunk)

            chunk = []

    chunk = []

    notes = []

del chunk

del notes
print(dataset[0])
dataset = np.array(dataset)

dataset.shape
import tensorflow as tf

from tensorflow import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Bidirectional

from keras.layers import LSTM, Reshape, RepeatVector, TimeDistributed

from sklearn.model_selection import train_test_split

from keras.layers.advanced_activations import LeakyReLU
dataset = dataset.reshape(len(dataset),16,1)

dataset.shape
noise = np.random.normal(0,1,(len(dataset),4,4))

noise.shape
#initialize model

model = Sequential()



#encoder model

model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(4, 4)))

model.add(LeakyReLU(alpha=0.2))



model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.3))   



model.add(Bidirectional(LSTM(128)))

model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.3))



#specifying output to have 16 timesteps

model.add(RepeatVector(16))



#decoder model

model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.3))



model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.3))   



model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.3))   



model.add(TimeDistributed(Dense(128)))

model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.3))

model.add(TimeDistributed(Dense(128)))

model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.3))

#specifying 1 feature as the output

model.add(TimeDistributed(Dense(1)))

model.add(LeakyReLU(alpha=0.2))



model.compile(loss='mean_squared_error', optimizer='adam')



model.summary()
#normalize note values to be between 0 and 1

scale = np.max(dataset)

dataset = dataset/scale

#splitting data into train and test sets. 3/4 train, 1/4 test.

x_train,x_test,y_train,y_test = train_test_split(noise, dataset, test_size=0.25, shuffle=True, random_state=42)
history = model.fit(x_train, y_train, epochs=25, batch_size=108, verbose=1,validation_data=(x_test, y_test))
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Test', 'Validation'], loc='upper right')

plt.show()
random = np.random.normal(0,1,(1,4,4))



predict = model.predict(random)



#adjusting from normalization

predict = predict * scale
print(predict)
midler = MidiFile()

track = MidiTrack()

midler.tracks.append(track)

track.append(Message('program_change', program=2, time=0))

for x in range(16):

    track.append(Message('note_on', note=int(predict[0][x][0]), velocity=64, time=20))

    track.append(Message('note_off', note=int(predict[0][x][0]), velocity=64, time=20))

    midler.save('new_song.mid')