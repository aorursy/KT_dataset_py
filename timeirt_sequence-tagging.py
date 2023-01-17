import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import json
import os
import re

def getMelodyString(doc):
    music_content = []
    for c1 in doc['children']:
        if 'children' not in c1: 
                continue
        for l1 in c1['children']:
            if 'children' not in l1: 
                continue
            for s1 in l1['children']:
                if 'notes' not in s1: 
                    continue
                if 'spaced' not in s1['notes']:
                    continue
                temp_syl = []
                for logic_seg in s1['notes']['spaced']:
                    
                    temp_logic_seg = []
                    for graph_seg in logic_seg['nonSpaced']:
                        temp_graph_seg = []
                        for note in graph_seg['grouped']:
                            temp_graph_seg.append(note['base']+str(note['octave']))
                        temp_logic_seg.append(temp_graph_seg)
                    temp_syl.append(temp_logic_seg)
                music_content.append(temp_syl)
    return music_content

def import_corpus(directory):
    files = []
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirname, filename))
    
    corpus = []
    for file in files:
        if not re.search("Apt 17",file):
            continue
        with open(file, 'r') as f:
            file_content = json.load(f)
        corpus.append(file_content)
    print(str(len(corpus))+ " files found")
    
    corpus = list(map(getMelodyString, corpus))
    corpus = [item for sublist in corpus for item in sublist]
    
    print(str(len(corpus)) + " syllables")
    
    return corpus


corpus = import_corpus('/kaggle/input')
np.save("corpus1.npy",corpus)
corpus[0]
from keras.preprocessing.sequence import pad_sequences
class Syllable:

    def __init__(self):
   
        self.note_to_id = [
            'E4','F3','G3','A3','B3','C4','D4','E4','F4','G4','A4','B4','C5','D5','E5','F5','G5','A5','B5',
        ]
        self.n_timesteps = 38
    
    def flatten(self, l):
        return [item for sublist in l for item in sublist]
    
    def encode_x(self, syl_x):
        encoded = []
        for note in syl_x:
            encoded.append([self.note_to_id.index(note)]) # shift to leave 0 values for masking
        return encoded
    def decode_x(self, syl_x):
        decoded = []
        for note in syl_x:
            decoded.append(self.note_to_id[0][note-1])
    
    def get_note_matrix(self, syllable_batch, min_length = 4):
        x = []
        y = []
        longest_seq = 0
        temp_seq = 0
        turn = 0
        for doc in syllable_batch:
            if len(self.flatten(doc)) < min_length:
                continue
            new_x_seq = []
            new_y_seq = []
            for ls in doc:
                temp_logic = []
                for gs in ls:
                    temp_graphic = []
                    for n in gs:
                        temp_graphic.append(n)
                        new_y_seq.append([1])
                        temp_seq += 1
                    if len(temp_graphic) == 1:
                        new_y_seq[-1] = [2]
                    else:
                        new_y_seq[-1] = [3]
                    temp_logic.extend(temp_graphic)
                firstElOfLS = -len(temp_logic)
                new_x_seq.extend(temp_logic)

            if temp_seq > longest_seq:
                longest_seq = temp_seq
            temp_seq = 0

            
            x.append(self.encode_x(new_x_seq))
            y.append(new_y_seq)
            turn += 1
        return pad_sequences(x, dtype='float32', padding='pre'), pad_sequences(y, dtype='float32', padding='pre')
        return x, y
    


s = Syllable()
X, y = s.get_note_matrix(corpus)

list(zip(X[0],y[0]))
X[0]


train_test_ratio = 0.7
train_valid_ratio = 0.8

thres_train_test = int(len(X)*train_test_ratio)
thres_train_valid = int(thres_train_test * train_valid_ratio)

train_X = X[0:thres_train_valid]
train_y = y[0:thres_train_valid]

test_X = X[thres_train_test:]
test_y = y[thres_train_test:]

valid_X = X[thres_train_valid:thres_train_test]
valid_y = y[thres_train_valid:thres_train_test]


from tensorflow.keras.layers import Masking, Dense, LSTM, Masking, Bidirectional, Dropout
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Masking(mask_value=0, input_shape=( 33, 1)))
model.add(Bidirectional(LSTM(120, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(17, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)


history = model.fit(train_X, train_y, epochs=100, batch_size=10, verbose=2, validation_data=(valid_X, valid_y), 
                    callbacks=[checkpoint])



i = 56
model.predict_classes(test_X[i:i+1], verbose=1)
model.save('model.h5')
test_y[i:i+1]
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
model.evaluate(test_X,test_y)
test_y[20:21]