#@title Install all dependencies and import all modules
!pip install pyFluidSynth
!apt install fluidsynth #Pip does not work for some reason. Only apt works
!pip install midi2audio
!pip install pretty_midi
!pip install pypianoroll
!pip install mir_eval
!pip install keras_self_attention
from midi2audio import FluidSynth
from google.colab import output
from IPython.display import display, Javascript, HTML, Audio


# Imports
from music21 import converter, instrument, note, chord, stream, midi
import glob
import time
import numpy as np
import keras.utils as utils
import pandas as pd
import tensorflow as tf
import os
#@title Define all functions and variables
# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128 # (stop playing all previous notes)
MELODY_NO_EVENT = 129 # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.

def streamToNoteArray(stream):
    """
    Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
        0-127 - note on at specified pitch
        128   - note off
        129   - no event
    """
    # Part one, extract from stream
    total_length = np.int(np.round(stream.flat.highestTime / 0.25)) # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append([np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.sortAscending().pitches[-1].midi])
    np_stream_list = np.array(stream_list, dtype=np.int)
    df = pd.DataFrame({'pos': np_stream_list.T[0], 'dur': np_stream_list.T[1], 'pitch': np_stream_list.T[2]})
    df = df.sort_values(['pos','pitch'], ascending=[True, False]) # sort the dataframe properly
    df = df.drop_duplicates(subset=['pos']) # drop duplicate values
    # part 2, convert into a sequence of note events
    output = np.zeros(total_length+1, dtype=np.int16) + np.int16(MELODY_NO_EVENT)  # set array full of no events by default.
    # Fill in the output list
    for i in range(total_length):
        if not df[df.pos==i].empty:
          try:
            n = df[df.pos==i].iloc[0] # pick the highest pitch at each semiquaver
            output[i] = n.pitch # set note on
            output[i+n.dur] = MELODY_NOTE_OFF
          except:
              print('Bad note. Skipping...')
    return output


def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df['offset'] = df.index
    df['duration'] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    df.duration = df.duration.diff(-1) * -1 * 0.25  # calculate durations and change to quarter note fractions
    df = df.fillna(0.25)
    return df[['code','duration']]


def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            new_note = note.Rest() # bit of an oversimplification, doesn't produce long notes.
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream

#wm_mid = converter.parse("/content/seed.mid")
#wm_mid.show()
#wm_mel_rnn = streamToNoteArray(wm_mid)
#print(wm_mel_rnn)
#noteArrayToStream(wm_mel_rnn)

#@title Alex Piano Only Drafts Original 1500 MIDIs 
%cd /content/Performance-RNN-PyTorch/dataset/midi
!wget 'https://github.com/asigalov61/AlexMIDIDataSet/raw/master/AlexMIDIDataSet-CC-BY-NC-SA-All-Drafts-Piano-Only.zip'
!unzip -j 'AlexMIDIDataSet-CC-BY-NC-SA-All-Drafts-Piano-Only.zip'
#@title Execute this cell to upload your MIDIs Data Set. Do not upload a lot and make sure that the files are not broken or have unusual configuration/settings.
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
#@title Parse the uploaded MIDI DataSet into a special Numpy Array of notes
import time
midi_files = glob.glob("/content/*.mid") # this won't work, no files there.

training_arrays = []
for f in midi_files:
    try:
        start = time.clock()
        s = converter.parse(f)
        print("Parsed:", f, "it took", time.clock() - start)
    except:
        continue
    for p in s.parts:
        start = time.clock()
        arr = streamToNoteArray(p)
        training_arrays.append(arr)
        print("Converted:", f, "it took", time.clock() - start)

training_dataset = np.array(training_arrays)
np.savez('melody_training_dataset.npz', train=training_dataset)
#@title Training Hyperparameters { run: "auto" }
generated_sequence_length = 128 #@param {type:"slider", min:0, max:256, step:8}
hidden_layer_size = 256 #@param {type:"slider", min:0, max:512, step:16}
number_of_training_epochs = 60 #@param {type:"slider", min:0, max:200, step:1}
training_batch_size = 2048 #@param {type:"number"}

VOCABULARY_SIZE = 130 # known 0-127 notes + 128 note_off + 129 no_event
SEQ_LEN = generated_sequence_length
BATCH_SIZE = training_batch_size
HIDDEN_UNITS = hidden_layer_size
EPOCHS = number_of_training_epochs
SEED = 2345  # 2345 seems to be good.
np.random.seed(SEED)

with np.load('./melody_training_dataset.npz', allow_pickle=True) as data:
    train_set = data['train']

print("Training melodies:", len(train_set))
#@title Defining additional Conversion Functions
def slice_sequence_examples(sequence, num_steps):
    """Slice a sequence into redundant sequences of lenght num_steps."""
    xs = []
    for i in range(len(sequence) - num_steps - 1):
        example = sequence[i: i + num_steps]
        xs.append(example)
    return xs

def seq_to_singleton_format(examples):
    """
    Return the examples in seq to singleton format.
    """
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs,ys)

# Prepare training data as X and Y.
# This slices the melodies into sequences of length SEQ_LEN+1.
# Then, each sequence is split into an X of length SEQ_LEN and a y of length 1.

# Slice the sequences:
slices = []
for seq in train_set:
    slices +=  slice_sequence_examples(seq, SEQ_LEN+1)

# Split the sequences into Xs and ys:
X, y = seq_to_singleton_format(slices)
# Convert into numpy arrays.
X = np.array(X)
y = np.array(y)

# Look at the size of the training corpus:
print("Total Training Corpus:")
print("X:", X.shape)
print("y:", y.shape)
print()

# Have a look at one example:
print("Looking at one example:")
print("X:", X[95])
print("y:", y[95])
# Note: Music data is sparser than text, there's lots of 129s (do nothing)
# and few examples of any particular note on.
# As a result, it's a bit harder to train a melody-rnn.
#@title Uploaded MIDIs Statitics
# Do some stats on the corpus.
all_notes = np.concatenate(train_set)
print("Number of notes:")
print(all_notes.shape)
all_notes_df = pd.DataFrame(all_notes)
print("Notes that do appear:")
unique, counts = np.unique(all_notes, return_counts=True)
print(unique)
print("Notes that don't appear:")
print(np.setdiff1d(np.arange(0,129),unique))

print("Plot the relative occurences of each note:")
import matplotlib.pyplot as plt
%matplotlib inline

#plt.style.use('dark_background')
plt.bar(unique, counts)
plt.yscale('log')
plt.xlabel('melody RNN value')
plt.ylabel('occurences (log scale)')
#@title Import needed modules and build the model
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout, Bidirectional, Flatten
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras_self_attention import SeqSelfAttention
# build the model: 2-layer LSTM network.
# Using Embedding layer and sparse_categorical_crossentropy loss function 
# to save some effort in preparing data.
print('Building model...')
model_train = Sequential()
model_train.add(Embedding(VOCABULARY_SIZE, HIDDEN_UNITS, input_length=SEQ_LEN))

# LSTM part
model_train.add(LSTM(HIDDEN_UNITS, return_sequences=True))
model_train.add(Dropout(0.3))
model_train.add(LSTM(HIDDEN_UNITS))
model_train.add(Dropout(0.3))
model_train.add(Flatten())
# Project back to vocabulary
model_train.add(Dense(VOCABULARY_SIZE, activation='softmax'))
model_train.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_train.summary()
#@title Train the model (this takes time, 50 epochs min recommended) and plot the results

import matplotlib.pyplot as plt

history = model_train.fit(X, y, validation_split=0.33, batch_size=BATCH_SIZE, epochs=EPOCHS)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model_train.save("noobiano-pre-trained-model.h5")
#@title SAVE
model_train.save("noobiano-pre-trained-model.h5")
#@title LOAD
# Load if necessary - don't need to do this.
model_train = keras.models.load_model("noobiano-pre-trained-model.h5")
#@title Build a decoding model (input length 1, batch size 1, stateful)
# Build a decoding model (input length 1, batch size 1, stateful)
model_dec = Sequential()
model_dec.add(Embedding(VOCABULARY_SIZE, HIDDEN_UNITS, input_length=1, batch_input_shape=(1,1)))
# LSTM part
model_dec.add(LSTM(HIDDEN_UNITS, stateful=True, return_sequences=True))
model_dec.add(Dropout(0.3))
model_dec.add(LSTM(HIDDEN_UNITS, stateful=True))
model_dec.add(Dropout(0.3))
model_dec.add(Flatten())
# project back to vocabulary
model_dec.add(Dense(VOCABULARY_SIZE, activation='softmax'))
model_dec.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model_dec.summary()
# set weights from training model
#model_dec.set_weights(model_train.get_weights())
model_dec.load_weights("noobiano-pre-trained-model.h5")
#@title Define Sampling/Generation Functions
def sample(preds, temperature=1.0):
    """ helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

## Sampling function

def sample_model(seed, model_name, length=400, temperature=1.0):
    '''Samples a musicRNN given a seed sequence.'''
    generated = []  
    generated.append(seed)
    next_index = seed
    for i in range(length):
        x = np.array([next_index])
        x = np.reshape(x,(1,1))
        preds = model_name.predict(x, verbose=0)[0]
        next_index = sample(preds, temperature)        
        generated.append(next_index)
    return np.array(generated)
#@title Generate some Music from your model :) Play with parameters below until you get what you like
primer_length = 16 #@param {type:"slider", min:1, max:128, step:1}
desired_composition_length_in_tokens = 512 #@param {type:"slider", min:0, max:1024, step:8}
creativity_temperature = 0.7 #@param {type:"slider", min:0, max:4, step:0.1}
model_dec.reset_states() # Start with LSTM state blank
o = sample_model(primer_length, model_dec, length=desired_composition_length_in_tokens, temperature=creativity_temperature) # generate 8 bars of melody

melody_stream = noteArrayToStream(o) # turn into a music21 stream
#melody_stream.show() # show the score.
fp = melody_stream.write('midi', fp='output_midi.mid')
#@title Plot and Graph the Output :)
graphs_length_inches = 18 #@param {type:"slider", min:0, max:20, step:1}
notes_graph_height = 6 #@param {type:"slider", min:0, max:20, step:1}
highest_displayed_pitch = 90 #@param {type:"slider", min:1, max:128, step:1}
lowest_displayed_pitch = 40 #@param {type:"slider", min:1, max:128, step:1}
pr_color_map = "Blues"
rendered_wav_graph_height = 3
import librosa
import numpy as np
import pretty_midi
import pypianoroll
from pypianoroll import Multitrack, Track
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('SVG')
# For plotting
import mir_eval.display
import librosa.display
%matplotlib inline


midi_data = pretty_midi.PrettyMIDI('/content/output_midi.mid')

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))



roll = np.zeros([int(graphs_length_inches), 128])
# Plot the output

#track = Multitrack('/content/output_midi.mid', name='track')
#plt.figure(figsize=[graphs_length_inches, notes_graph_height])
#fig, ax = track.plot()
#fig.set_size_inches(graphs_length_inches, notes_graph_height)
plt.figure(figsize=[graphs_length_inches, notes_graph_height])
ax2 = plot_piano_roll(midi_data, lowest_displayed_pitch, highest_displayed_pitch)
plt.show(block=False)

## Play a melody stream


!cp /usr/share/sounds/sf2/FluidR3_GM.sf2 /content/font.sf2


FluidSynth("/content/font.sf2").midi_to_audio('output_midi.mid','output_wav.wav')
# set the src and play
Audio("output_wav.wav")


from google.colab import drive
drive.mount('/content/drive')