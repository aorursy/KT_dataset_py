#@title Install dependencies
!pip install pyknon
!pip install pretty_midi
!pip install pypianoroll
!pip install mir_eval
!apt install fluidsynth #Pip does not work for some reason. Only apt works
!pip install midi2audio
!cp /usr/share/sounds/sf2/FluidR3_GM.sf2 /content/font.sf2
#@title Load all modules, check the available devices (GPU/CPU), and setup MIDI parameters
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import keras
from keras.utils import to_categorical

import time

import pretty_midi
from midi2audio import FluidSynth
from google.colab import output
from IPython.display import display, Javascript, HTML, Audio

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print('Available Device:', device)

!mkdir /content/midis

sample_freq_variable = 12 #@param {type:"number"}
note_range_variable = 62 #@param {type:"number"}
note_offset_variable = 33 #@param {type:"number"}
number_of_instruments = 2 #@param {type:"number"}
chamber_option = True #@param {type:"boolean"}
#@title (OPTION 1) Convert your own MIDIs to Notewise TXT DataSet (before running this cell, upload your MIDI DataSet to /content/midis folder)
import tqdm.auto
import argparse
import random
import os
import numpy as np
from math import floor
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note
import music21
from music21 import instrument, volume
from music21 import midi as midiModule
from pathlib import Path
import glob, sys
from music21 import converter, instrument
%cd /content
notes=[]
InstrumentID=0
folder = '/content/midis/*mid'
for file in tqdm.auto.tqdm(glob.glob(folder)):
    filename = file[-53:]
    print(filename)

    # fname = "../midi-files/mozart/sonat-3.mid"
    fname = filename

    mf=music21.midi.MidiFile()
    mf.open(fname)
    mf.read()
    mf.close()
    midi_stream=music21.midi.translate.midiFileToStream(mf)
    midi_stream



    sample_freq=sample_freq_variable
    note_range=note_range_variable
    note_offset=note_offset_variable
    chamber=chamber_option
    numInstruments=number_of_instruments

    s = midi_stream
    #print(s.duration.quarterLength)

    s[0].elements


    maxTimeStep = floor(s.duration.quarterLength * sample_freq)+1
    score_arr = np.zeros((maxTimeStep, numInstruments, note_range))

    #print(maxTimeStep, "\n", score_arr.shape)

    # define two types of filters because notes and chords have different structures for storing their data
    # chord have an extra layer because it consist of multiple notes

    noteFilter=music21.stream.filters.ClassFilter('Note')
    chordFilter=music21.stream.filters.ClassFilter('Chord')
      

    # pitch.midi-note_offset: pitch is the numerical representation of a note. 
    #                         note_offset is the the pitch relative to a zero mark. eg. B-=25, C=27, A=24

    # n.offset: the timestamps of each note, relative to the start of the score
    #           by multiplying with the sample_freq, you make all the timestamps integers

    # n.duration.quarterLength: the duration of that note as a float eg. quarter note = 0.25, half note = 0.5
    #                           multiply by sample_freq to represent duration in terms of timesteps

    notes = []
    instrumentID = 0
    parts = instrument.partitionByInstrument(s)
    for i in range(len(parts.parts)): 
      instru = parts.parts[i].getInstrument()
      

    for n in s.recurse().addFilter(noteFilter):
        if chamber:
          # assign_instrument where 0 means piano-like and 1 means violin-like, and -1 means neither
          if instru.instrumentName == 'Violin':
            notes.append((n.pitch.midi-note_offset, floor(n.offset*sample_freq), 
              floor(n.duration.quarterLength*sample_freq), 1))
            
        notes.append((n.pitch.midi-note_offset, floor(n.offset*sample_freq), 
              floor(n.duration.quarterLength*sample_freq), 0))
        
    #print(len(notes))
    notes[-5:]

    # do the same using a chord filter

    for c in s.recurse().addFilter(chordFilter):
        # unlike the noteFilter, this line of code is necessary as there are multiple notes in each chord
        # pitchesInChord is a list of notes at each chord eg. (<music21.pitch.Pitch D5>, <music21.pitch.Pitch F5>)
        pitchesInChord=c.pitches
        
        # do same as noteFilter and append all notes to the notes list
        for p in pitchesInChord:
            notes.append((p.midi-note_offset, floor(c.offset*sample_freq), 
                          floor(c.duration.quarterLength*sample_freq), 1))

        # do same as noteFilter and append all notes to the notes list
        for p in pitchesInChord:
            notes.append((p.midi-note_offset, floor(c.offset*sample_freq), 
                          floor(c.duration.quarterLength*sample_freq), 0))
    #print(len(notes))
    notes[-5:]

    # the variable/list "notes" is a collection of all the notes in the song, not ordered in any significant way

    for n in notes:
        
        # pitch is the first variable in n, previously obtained by n.midi-note_offset
        pitch=n[0]
        
        # do some calibration for notes that fall our of note range
        # i.e. less than 0 or more than note_range
        while pitch<0:
            pitch+=12
        while pitch>=note_range:
            pitch-=12
            
        # 3rd element refers to instrument type => if instrument is violin, use different pitch calibration
        if n[3]==1:      #Violin lowest note is v22
            while pitch<22:
                pitch+=12

        # start building the 3D-tensor of shape: (796, 1, 38)
        # score_arr[0] = timestep
        # score_arr[1] = type of instrument
        # score_arr[2] = pitch/note out of the range of note eg. 38
        
        # n[0] = pitch
        # n[1] = timestep
        # n[2] = duration
        # n[3] = instrument
        #print(n[3])
        score_arr[n[1], n[3], pitch]=1                  # Strike note
        score_arr[n[1]+1:n[1]+n[2], n[3], pitch]=2      # Continue holding note

    #print(score_arr.shape)
    # print first 5 timesteps
    score_arr[:5,0,]


    for timestep in score_arr:
        #print(list(reversed(range(len(timestep)))))
        break

    instr={}
    instr[0]="p"
    instr[1]="v"

    score_string_arr=[]

    # loop through all timesteps
    for timestep in score_arr:
        
        # selecting the instruments: i=0 means piano and i=1 means violin
        for i in list(reversed(range(len(timestep)))):   # List violin note first, then piano note
            
            # 
            score_string_arr.append(instr[i]+''.join([str(int(note)) for note in timestep[i]]))

    #print(type(score_string_arr), len(score_string_arr))
    score_string_arr[:5]

    modulated=[]
    # get the note range from the array
    note_range=len(score_string_arr[0])-1

    for i in range(0,12):
        for chord in score_string_arr:
            
            # minus the instrument letter eg. 'p'
            # add 6 zeros on each side of the string
            padded='000000'+chord[1:]+'000000'
            
            # add back the instrument letter eg. 'p'
            # append window of len=note_range back into 
            # eg. if we have "00012345000"
            # iteratively, we want to get "p00012", "p00123", "p01234", "p12345", "p23450", "p34500", "p45000",
            modulated.append(chord[0]+padded[i:i+note_range])

    # 796 * 12
    #print(len(modulated))
    modulated[:5]

    # input of this function is a modulated string
    long_string = modulated

    translated_list=[]

    # for every timestep of the string
    for j in range(len(long_string)):
        
        # chord at timestep j eg. 'p00000000000000000000000000000000000100'
        chord=long_string[j]
        next_chord=""
        
        # range is from next_timestep to max_timestep
        for k in range(j+1, len(long_string)):
            
            # checking if instrument of next chord is same as current chord
            if long_string[k][0]==chord[0]:
                
                # if same, set next chord as next element in modulation
                # otherwise, keep going until you find a chord with the same instrument
                # when you do, set it as the next chord
                next_chord=long_string[k]
                break
        
        # set prefix as the instrument
        # set chord and next_chord to be without the instrument prefix
        # next_chord is necessary to check when notes end
        prefix=chord[0]
        chord=chord[1:]
        next_chord=next_chord[1:]
        
        # checking for non-zero notes at one particular timestep
        # i is an integer indicating the index of each note the chord
        for i in range(len(chord)):
            
            if chord[i]=="0":
                continue
            
            # set note as 2 elements: instrument and index of note
            # examples: p22, p16, p4
            #p = music21.pitch.Pitch()
            #nt = music21.note.Note(p)
            #n.volume.velocity = 20
            #nt.volume.client == nt
            #V = nt.volume.velocity
            #print(V)
            #note=prefix+str(i)+' V' + str(V)
            note=prefix+str(i)                
            
            # if note in chord is 1, then append the note eg. p22 to the list
            if chord[i]=="1":
                translated_list.append(note)
            
            # If chord[i]=="2" do nothing - we're continuing to hold the note
            
            # unless next_chord[i] is back to "0" and it's time to end the note.
            if next_chord=="" or next_chord[i]=="0":      
                translated_list.append("end"+note)

        # wait indicates end of every timestep
        if prefix=="p":
            translated_list.append("wait")

    #print(len(translated_list))
    translated_list[:10]

    # this section transforms the list of notes into a string of notes

    # initialize i as zero and empty string
    i=0
    translated_string=""


    while i<len(translated_list):
        
        # stack all the repeated waits together using an integer to indicate the no. of waits
        # eg. "wait wait" => "wait2"
        wait_count=1
        if translated_list[i]=='wait':
            while wait_count<=sample_freq*2 and i+wait_count<len(translated_list) and translated_list[i+wait_count]=='wait':
                wait_count+=1
            translated_list[i]='wait'+str(wait_count)
            
        # add next note
        translated_string+=translated_list[i]+" "
        i+=wait_count

    translated_string[:100]
    len(translated_string)

    #print("chordwise encoding type and length:", type(modulated), len(modulated))
    #print("notewise encoding type and length:", type(translated_string), len(translated_string))

    # default settings: sample_freq=12, note_range=62

    chordwise_folder = "../"
    notewise_folder = "../"

    # export chordwise encoding
#    f=open(chordwise_folder+fname+"_chordwise"+".txt","w+")
#    f.write(" ".join(modulated))
#    f.close()

    # export notewise encoding
    f=open(notewise_folder+fname+"_notewise"+".txt","w+")
    f.write(translated_string)
    f.close()

folder = '/content/midis/*notewise.txt'


filenames = glob.glob('/content')
with open('notewise_custom_dataset.txt', 'w') as outfile:
    for fname in glob.glob(folder)[-53:]:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

#folder = '/content/midis/*chordwise.txt'

#filenames = glob.glob('/content')
#with open('chordwise_custom_dataset.txt', 'w') as outfile:
#    for fname in glob.glob(folder)[-53:]:
#        with open(fname) as infile:
#            for line in infile:
#                outfile.write(line)
#@title (OPTION 2) Download ready-to-use Piano and Chamber Notewise DataSets
%cd /content/
!wget 'https://github.com/asigalov61/SuperPiano/raw/master/Super%20Chamber%20Piano%20Violin%20Notewise%20DataSet.zip'
!unzip '/content/Super Chamber Piano Violin Notewise DataSet.zip'
!rm '/content/Super Chamber Piano Violin Notewise DataSet.zip'

!wget 'https://github.com/asigalov61/SuperPiano/raw/master/Super%20Chamber%20Piano%20Only%20Notewise%20DataSet.zip'
!unzip '/content/Super Chamber Piano Only Notewise DataSet.zip'
!rm '/content/Super Chamber Piano Only Notewise DataSet.zip'
#@title Load and Encode TXT Notes DataSet
select_training_dataset_file = "/content/notewise_custom_dataset.txt" #@param {type:"string"}

# replace with any text file containing full set of data
MIDI_data = select_training_dataset_file

with open(MIDI_data, 'r') as file:
    text = file.read()

# get vocabulary set
words = sorted(tuple(set(text.split())))
n = len(words)

# create word-integer encoder/decoder
word2int = dict(zip(words, list(range(n))))
int2word = dict(zip(list(range(n)), words))

# encode all words in dataset into integers
encoded = np.array([word2int[word] for word in text.split()])
#@title Define all functions
# define model using the pytorch nn module
class WordLSTM(nn.ModuleList):
    
    def __init__(self, sequence_len, vocab_size, hidden_dim, batch_size):
        super(WordLSTM, self).__init__()
        
        # init the hyperparameters
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        
        # first layer lstm cell
        self.lstm_1 = nn.LSTMCell(input_size=vocab_size, hidden_size=hidden_dim)
        
        # second layer lstm cell
        self.lstm_2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # third layer lstm cell
        #self.lstm_3 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # dropout layer
        self.dropout = nn.Dropout(p=0.35)
        
        # fully connected layer
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        
    # forward pass in training   
    def forward(self, x, hc):
        """
            accepts 2 arguments: 
            1. x: input of each batch 
                - shape 128*149 (batch_size*vocab_size)
            2. hc: tuple of init hidden, cell states 
                - each of shape 128*512 (batch_size*hidden_dim)
        """
        
        # create empty output seq
        output_seq = torch.empty((self.sequence_len,
                                  self.batch_size,
                                  self.vocab_size))
        # if using gpu        
        output_seq = output_seq.to(device)
        
        # init hidden, cell states for lstm layers
        hc_1, hc_2, hc_3 = hc, hc, hc
        
        # for t-th word in every sequence 
        for t in range(self.sequence_len):
            
            # layer 1 lstm
            hc_1 = self.lstm_1(x[t], hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3
            
            # dropout and fully connected layer
            output_seq[t] = self.fc(self.dropout(h_2))
            
        return output_seq.view((self.sequence_len * self.batch_size, -1))
          
    def init_hidden(self):
        
        # initialize hidden, cell states for training
        # if using gpu
        return (torch.zeros(self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.batch_size, self.hidden_dim).to(device))
    
    def init_hidden_generator(self):
        
        # initialize hidden, cell states for prediction of 1 sequence
        # if using gpu
        return (torch.zeros(1, self.hidden_dim).to('cpu'),
                torch.zeros(1, self.hidden_dim).to('cpu'))
    
    def predict(self, seed_seq, top_k=5, pred_len=128):
        """
            accepts 3 arguments: 
            1. seed_seq: seed string sequence for prediction (prompt)
            2. top_k: top k words to sample prediction from
            3. pred_len: number of words to generate after the seed seq
        """
        
        # set evaluation mode
        self.eval()
        
        # split string into list of words
        seed_seq = seed_seq.split()
        
        # get seed sequence length
        seed_len = len(seed_seq)
        
        # create output sequence
        out_seq = np.empty(seed_len+pred_len)
        
        # append input seq to output seq
        out_seq[:seed_len] = np.array([word2int[word] for word in seed_seq])
 
        # init hidden, cell states for generation
        hc = self.init_hidden_generator()
        hc_1, hc_2, hc_3 = hc, hc, hc
        
        # feed seed string into lstm
        # get the hidden state set up
        for word in seed_seq[:-1]:
            
            # encode starting word to one-hot encoding
            word = to_categorical(word2int[word], num_classes=self.vocab_size)

            # add batch dimension
            word = torch.from_numpy(word).unsqueeze(0)
            # if using gpu
            word = word.to('cpu') 
            
            # layer 1 lstm
            hc_1 = self.lstm_1(word, hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3            

        word = seed_seq[-1]
        
        # encode starting word to one-hot encoding
        word = to_categorical(word2int[word], num_classes=self.vocab_size)

        # add batch dimension
        word = torch.from_numpy(word).unsqueeze(0)
        # if using gpu
        word = word.to('cpu') 

        # forward pass
        for t in range(pred_len):
            
            # layer 1 lstm
            hc_1 = self.lstm_1(word, hc_1)
            h_1, c_1 = hc_1
            
            # layer 2 lstm
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, c_2 = hc_2

            # layer 3 lstm
            #hc_3 = self.lstm_3(h_2, hc_3)
            #h_3, c_3 = hc_3
            
            # fully connected layer without dropout (no need)
            output = self.fc(h_2)
            
            # software to get probabilities of output options
            output = F.softmax(output, dim=1)
            
            # get top k words and corresponding probabilities
            p, top_word = output.topk(top_k)
            # if using gpu           
            p = p.cpu()
            
            # sample from top k words to get next word
            p = p.detach().squeeze().numpy()
            top_word = torch.squeeze(top_word)
            
            word = np.random.choice(top_word, p = p/p.sum())
            
            # add word to sequence
            out_seq[seed_len+t] = word
            
            # encode predicted word to one-hot encoding for next step
            word = to_categorical(word, num_classes=self.vocab_size)
            word = torch.from_numpy(word).unsqueeze(0)
            # if using gpu
            word = word.to('cpu')
            
        return out_seq


def get_batches(arr, n_seqs, n_words):
    """
        create generator object that returns batches of input (x) and target (y).
        x of each batch has shape 128*128*149 (batch_size*seq_len*vocab_size).
        
        accepts 3 arguments:
        1. arr: array of words from text data
        2. n_seq: number of sequence in each batch (aka batch_size)
        3. n_word: number of words in each sequence
    """
    
    # compute total elements / dimension of each batch
    batch_total = n_seqs * n_words
    
    # compute total number of complete batches
    n_batches = arr.size//batch_total
    
    # chop array at the last full batch
    arr = arr[: n_batches* batch_total]
    
    # reshape array to matrix with rows = no. of seq in one batch
    arr = arr.reshape((n_seqs, -1))
    
    # for each n_words in every row of the dataset
    for n in range(0, arr.shape[1], n_words):
        
        # chop it vertically, to get the input sequences
        x = arr[:, n:n+n_words]
        
        # init y - target with shape same as x
        y = np.zeros_like(x)
        
        # targets obtained by shifting by one
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, n+n_words]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        
        # yield function is like return, but creates a generator object
        yield x, y   
#@title Compile the Model
training_batch_size = 1024 #@param {type:"slider", min:0, max:1024, step:16}
attention_span_in_tokens = 256 #@param {type:"slider", min:0, max:512, step:64}
hidden_dimension_size = 256 #@param {type:"slider", min:0, max:512, step:64}
test_validation_ratio = 0.1 #@param {type:"slider", min:0, max:1, step:0.1}
learning_rate = 0.001 #@param {type:"number"}


# compile the network - sequence_len, vocab_size, hidden_dim, batch_size
net = WordLSTM(sequence_len=attention_span_in_tokens, vocab_size=len(word2int), hidden_dim=hidden_dimension_size, batch_size=training_batch_size)
# if using gpu
net.to(device)

# define the loss and the optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# split dataset into 90% train and 10% using index
val_idx = int(len(encoded) * (1 - test_validation_ratio))
train_data, val_data = encoded[:val_idx], encoded[val_idx:]

# empty list for the validation losses
val_losses = list()

# empty list for the samples
samples = list()
#@title (OPTION 1) Train the Model
number_of_training_epochs = 300 #@param {type:"slider", min:1, max:300, step:1}

import tqdm

# track time
start_time = time.time()

# declare seed sequence
#seed_string = "p47 p50 wait8 endp47 endp50 wait4 p47 p50 wait8 endp47 endp50"

# finally train the model
for epoch in tqdm.auto.tqdm(range(number_of_training_epochs)):
    
    # init the hidden and cell states to zero
    hc = net.init_hidden()
    
    # (x, y) refers to one batch with index i, where x is input, y is target
    for i, (x, y) in enumerate(get_batches(train_data, training_batch_size, hidden_dimension_size)):
        
        # get the torch tensors from the one-hot of training data
        # also transpose the axis for the training set and the targets
        x_train = torch.from_numpy(to_categorical(x, num_classes=net.vocab_size).transpose([1, 0, 2]))
        targets = torch.from_numpy(y.T).type(torch.LongTensor)  # tensor of the target
        
        # if using gpu
        x_train = x_train.to(device)
        targets = targets.to(device)
        
        # zero out the gradients
        optimizer.zero_grad()
        
        # get the output sequence from the input and the initial hidden and cell states
        # calls forward function
        output = net(x_train, hc)
    
        # calculate the loss
        # we need to calculate the loss across all batches, so we have to flat the targets tensor
        loss = criterion(output, targets.contiguous().view(training_batch_size*hidden_dimension_size))
        
        # calculate the gradients
        loss.backward()
        
        # update the parameters of the model
        optimizer.step()
        
        # track time
    
        # feedback every 100 batches
        if i % 100 == 0:
            
            # initialize the validation hidden state and cell state
            val_h, val_c = net.init_hidden()
            
            for val_x, val_y in get_batches(val_data, training_batch_size, hidden_dimension_size):
        
                # prepare the validation inputs and targets
                val_x = torch.from_numpy(to_categorical(val_x).transpose([1, 0, 2]))
                val_y = torch.from_numpy(val_y.T).type(torch.LongTensor).contiguous().view(training_batch_size*hidden_dimension_size)
  
                # if using gpu
                val_x = val_x.to(device)
                val_y = val_y.to(device)
            
                # get the validation output
                #val_output = net(val_x, (val_h, val_c))
                
                # get the validation loss
                #val_loss = criterion(val_output, val_y)
                
                # append the validation loss
                #val_losses.append(val_loss.item())
                 
                # samples.append(''.join([int2char[int_] for int_ in net.predict("p33", seq_len=1024)]))
                
#            with open("../content" + str(epoch) + "_batch" + str(i) + ".txt", "w") as loss_file:
#                loss_file.write("Epoch: {}, Batch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}".format(epoch, i, loss.item(), val_loss.item()))

#            with open("../content" + str(epoch) + "_batch" + str(i) + ".txt", "w") as outfile:
#                outfile.write(' '.join([int2word[int_] for int_ in net.predict(seed_seq=seed_string, pred_len=512)]))
        
            # track time
            duration = round(time.time() - start_time, 1)
            start_time = time.time()
    
            print("Epoch: {}, Batch: {}, Duration: {} sec, Test Loss: {}".format(epoch, i, duration, loss.item()))

#@title (OPTION 1) Save the trained Model from memory
torch.save(net, '/content/trained_model.h5')
#@title (OPTION 2) Download Super Chamber Piano Pre-Trained Chamber Model
%cd /content/
!wget 'https://github.com/asigalov61/SuperPiano/raw/master/trained_model.h5'
#@title (OPTION 2) Load existing/pre-trained Model checkpoint
model = torch.load('../content/trained_model.h5', map_location='cpu')
model.eval()
#@title Generate TXT and MIDI file
seed_prompt = "p24 v24" #@param {type:"string"}
tokens_to_generate = 512 #@param {type:"slider", min:0, max:1024, step:16}
time_coefficient = 1.25 #@param {type:"number"}
top_k_coefficient =  4#@param {type:"integer"}

with open("../content/output.txt", "w") as outfile:
    outfile.write(' '.join([int2word[int_] for int_ in model.predict(seed_seq=seed_prompt, pred_len=tokens_to_generate, top_k=top_k_coefficient)]))
import tqdm
import os
import dill as pickle
from pathlib import Path
import random
import numpy as np
import pandas as pd
from math import floor
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note
import music21
import random
import os, argparse

# default settings: sample_freq=12, note_range=62

def decoder(filename):
    
    filedir = '/content/'

    notetxt = filedir + filename

    with open(notetxt, 'r') as file:
        notestring=file.read()

    score_note = notestring.split(" ")

    # define some parameters (from encoding script)
    sample_freq=sample_freq_variable
    note_range=note_range_variable
    note_offset=note_offset_variable
    chamber=chamber_option
    numInstruments=number_of_instruments

    # define variables and lists needed for chord decoding
    speed=time_coefficient/sample_freq
    piano_notes=[]
    violin_notes=[]
    time_offset=0

    # start decoding here
    score = score_note

    i=0

    # for outlier cases, not seen in sonat-1.txt
    # not exactly sure what scores would have "p_octave_" or "eoc" (end of chord?)
    # it seems to insert new notes to the score whenever these conditions are met
    while i<len(score):
        if score[i][:9]=="p_octave_":
            add_wait=""
            if score[i][-3:]=="eoc":
                add_wait="eoc"
                score[i]=score[i][:-3]
            this_note=score[i][9:]
            score[i]="p"+this_note
            score.insert(i+1, "p"+str(int(this_note)+12)+add_wait)
            i+=1
        i+=1


    # loop through every event in the score
    for i in tqdm.auto.tqdm(range(len(score))):

        # if the event is a blank, space, "eos" or unknown, skip and go to next event
        if score[i] in ["", " ", "<eos>", "<unk>"]:
            continue

        # if the event starts with 'end' indicating an end of note
        elif score[i][:3]=="end":

            # if the event additionally ends with eoc, increare the time offset by 1
            if score[i][-3:]=="eoc":
                time_offset+=1
            continue

        # if the event is wait, increase the timestamp by the number after the "wait"
        elif score[i][:4]=="wait":
            time_offset+=int(score[i][4:])
            continue

        # in this block, we are looking for notes   
        else:
            # Look ahead to see if an end<noteid> was generated
            # soon after.  
            duration=1
            has_end=False
            note_string_len = len(score[i])
            for j in range(1,200):
                if i+j==len(score):
                    break
                if score[i+j][:4]=="wait":
                    duration+=int(score[i+j][4:])
                if score[i+j][:3+note_string_len]=="end"+score[i] or score[i+j][:note_string_len]==score[i]:
                    has_end=True
                    break
                if score[i+j][-3:]=="eoc":
                    duration+=1

            if not has_end:
                duration=12

            add_wait = 0
            if score[i][-3:]=="eoc":
                score[i]=score[i][:-3]
                add_wait = 1

            try: 
                new_note=music21.note.Note(int(score[i][1:])+note_offset)    
                new_note.duration = music21.duration.Duration(duration*speed)
                new_note.offset=time_offset*speed
                if score[i][0]=="v":
                    violin_notes.append(new_note)
                else:
                    piano_notes.append(new_note)                
            except:
                print("Unknown note: " + score[i])




            time_offset+=add_wait

    # list of all notes for each instrument should be ready at this stage

    # creating music21 instrument objects      
    
    piano=music21.instrument.fromString("Piano")
    violin=music21.instrument.fromString("Violin")

    # insert instrument object to start (0 index) of notes list
    
    piano_notes.insert(0, piano)
    violin_notes.insert(0, violin)
    # create music21 stream object for individual instruments
    
    piano_stream=music21.stream.Stream(piano_notes)
    violin_stream=music21.stream.Stream(violin_notes)
    # merge both stream objects into a single stream of 2 instruments
    note_stream = music21.stream.Stream([piano_stream, violin_stream])

    
    note_stream.write('midi', fp="/content/"+filename[:-4]+".mid")
    print("Done! Decoded midi file saved to 'content/'")

    
decoder('output.txt')
from google.colab import files
files.download('/content/output.mid')
#@title Plot, Graph, and Listen to the Output :)
graphs_length_inches = 18 #@param {type:"slider", min:0, max:20, step:1}
notes_graph_height = 6 #@param {type:"slider", min:0, max:20, step:1}
highest_displayed_pitch = 92 #@param {type:"slider", min:1, max:128, step:1}
lowest_displayed_pitch = 24 #@param {type:"slider", min:1, max:128, step:1}

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


midi_data = pretty_midi.PrettyMIDI('/content/output.mid')

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))



roll = np.zeros([int(graphs_length_inches), 128])
# Plot the output

track = Multitrack('/content/output.mid', name='track')
plt.figure(figsize=[graphs_length_inches, notes_graph_height])
fig, ax = track.plot()
fig.set_size_inches(graphs_length_inches, notes_graph_height)
plt.figure(figsize=[graphs_length_inches, notes_graph_height])
ax2 = plot_piano_roll(midi_data, int(lowest_displayed_pitch), int(highest_displayed_pitch))
plt.show(block=False)


FluidSynth("/content/font.sf2", 16000).midi_to_audio('/content/output.mid', '/content/output.wav')
Audio('/content/output.wav', rate=16000)
from google.colab import drive
drive.mount('/content/drive')