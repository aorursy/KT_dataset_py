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
!pip install music21

from music21 import *
import glob
import pickle
import joblib
from collections import Counter
from matplotlib import pyplot as plt

# def read_midi(file):
    
#     print("Loading Music File:",file)
    
#     notes=[]
#     notes_to_parse = None
    
#     #parsing a midi file
#     midi = converter.parse(file)
  
#     #grouping based on different instruments
#     s2 = instrument.partitionByInstrument(midi)
#     print(type(s2))

#     #Looping over all the instruments
#     for part in s2.parts:
#         #select elements of only piano
#         if 'Piano' in str(part): 
        
#             notes_to_parse = part.recurse() 
      
#             #finding whether a particular element is note or a chord
#             for element in notes_to_parse:
                
#                 #note
#                 if isinstance(element, note.Note):
#                     notes.append(str(element.pitch))
                
#                 #chord
#                 elif isinstance(element, chord.Chord):
#                     notes.append('.'.join(str(n) for n in element.normalOrder))

#     return np.array(notes)
def read_midi(file):
    
    print("Loading Music File:",file)
    
    notes=[]
    notes_to_parse = None
    
    #parsing a midi file
    midi = converter.parse(file)
  
    #grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    #Looping over all the instruments
    for part in s2.parts:
        #select elements of only piano
#         if 'Piano' in str(part): 
        
        notes_to_parse = part.recurse() 

        #finding whether a particular element is note or a chord
        for element in notes_to_parse:

            #note
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            #chord
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)


def convert_to_midi(prediction_output,name):
   
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                
                cn=int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        # pattern is a note
        else:
            
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=name+'.mid')
# name2note = {}
# for file in glob.glob('/kaggle/input/anime-music-midi/*'):
#     print(file)
#     try:
#         file_name = file.split('/')[-1].split('.')[0]
#         res = read_midi(file)
#         name2note[file_name] = res
#     except:
#         pass
# I have saved name2note 
name2note = joblib.load('/kaggle/input/name2note/name2note.jb')
name2note_reg = {x:['.'.join(sound.split('.')[:2]) for sound in name2note[x]] for x in name2note}
all_notes = [x for sound in name2note_reg.values() for x in sound]
c = Counter(all_notes)
c.most_common()

id2notes= {i+1:x for i,x in enumerate(list({note for note in all_notes}))}
notes2id = {x:i for i,x in id2notes.items()}
music = list(name2note_reg.values())
ids = [[notes2id[x] for x in song] for song in music]
max_len = max([len(x) for x in ids])
ids_pad = [x[:max_len]+[0]*max(0,max_len-len(x)) for x in ids]
inputs_x = [x[:-1] for x in ids_pad]
inputs_y = [x[1:] for x in ids_pad]


import tensorflow as tf
# from tensorflow.data import Dataset
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy

notes_number = len(id2notes)+1

music_maker = Sequential()
music_maker.add(Embedding(notes_number, 100,mask_zero=True))
music_maker.add(LSTM(256, return_sequences=True))
music_maker.add(Dense(notes_number))
music_maker.compile(
    optimizer='Adam', loss=SparseCategoricalCrossentropy(from_logits=True))


music_maker.fit(x=inputs_x,
                y=inputs_y, epochs=100,batch_size=8)

import random
musics = []
for tone in random.sample(list(id2notes.keys()),2):
    last_tone = ''
    tones = str(tone)
    while tone != 0 and len(tones)<500:
        pred = music_maker.predict([tone])
        tone_candidates = tf.argsort(pred, -1,direction='DESCENDING')
        for x in tone_candidates[0][0][:3]:
            tone = int(x)
            if tones[-5:]+'|'+str(tone) not in tones:
#             last_tone != str(tone)  and last_tone+'|'+str(tone) not in tones[-100:] and 
            
                break
            tone = None
            
        if not tone:
            tone = int(random.choice(tone_candidates[0][0][:5]))
        last_tone = str(tone)
        
        tones += '|'+str(tone)
    musics.append([int(x) for x in tones.split('|')])
    music_maker.reset_states()
musics[1]
for i,tones in enumerate(musics):
    gen_music = [id2notes[x] for x in tones]
    convert_to_midi(gen_music,'gen_'+str(i))
starts = [x[0] for x in inputs_x]