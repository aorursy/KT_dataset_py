# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
input_dir = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        input_dir.append(os.path.join(dirname, filename))# Data directory names.

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Activation
import keras.utils as kutils
#col = ['', 'character', 'dialogue']
def get_df_from_str(df_line):
    data_vec = []
    in_commas = False
    word = ''
    for c in df_line:
        if c == '"':
            in_commas = not(in_commas)
            
        if in_commas and c!='"':
            word = word+c
        else:
            if len(word)>0:
                data_vec.append(word)
            word = ''
    
    #print(data_vec)
    return data_vec
        
    
    
def txt_to_df(txt):
    file = open(txt)
    char_diag = file.read()
    cp = 0
    header = True
    df = pd.DataFrame(columns=['character', 'dialogue'])
    while header:
        if char_diag[cp:cp+1] == '\n':
            header = False
        cp = cp+1
    #print(char_diag[cp:])
    
    while cp< len(char_diag):
        cp_start_line = cp
        header = True
        while header:
            if char_diag[cp:cp+1] == '\n':
                header = False
            cp = cp+1
        data_vec = get_df_from_str(char_diag[cp_start_line:cp])
        df = df.append({'character': data_vec[1], 'dialogue': data_vec[2]}, ignore_index=True)
     
    return df
### Load all data in a df
df_sw = txt_to_df(input_dir[0])
df_sw.append(txt_to_df(input_dir[1]), ignore_index=True )
df_sw.append(txt_to_df(input_dir[2]), ignore_index=True )
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import string

all_sents = [[w.lower() for w in word_tokenize(sen) if not w in string.punctuation] \
             for sen in df_sw['dialogue']]

x = []
y = []

print(all_sents[:10])

for sen in all_sents:
    for i in range(1, len(sen)):
        x.append(sen[:i])
        y.append(sen[i])
        

print(x[:10])
print(y[:10])
from sklearn.model_selection import train_test_split
import numpy as np

all_text = [c for sen in x for c in sen]
all_text += [c for c in y]

all_text.append('UNK') # Palavra desconhecida

words = list(set(all_text))
        
word_indexes = {word: index for index, word in enumerate(words)}      

max_features = len(word_indexes)

x = [[word_indexes[c] for c in sen] for sen in x]
y = [word_indexes[c] for c in y]

print(x[:10])
print(y[:10])

y = kutils.to_categorical(y, num_classes=max_features)

maxlen = max([len(sen) for sen in x])

print(maxlen)
x = pad_sequences(x, maxlen=maxlen)
x = pad_sequences(x, maxlen=maxlen)

print(x[:10,-10:])
print(y[:10,-10:])
embedding_size = 10

model = Sequential()
    
# Add Input Embedding Layer
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    
# Add Hidden Layer 1 - LSTM Layer
model.add(LSTM(100, dropout = 0.1 ))

# Add Hidden Layer 2 - LSTM Layer
model.add(Dense(50))

model.add(Dropout(0.1))


    
# Add Output Layer
model.add(Dense(max_features, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()
model.fit(x, y, epochs=20, verbose=5)
def get_word_by_index(index, word_indexes):
    for w, i in word_indexes.items():
        if index == i:
            return w
        
    return None

sample_seed = input()
response_seed = ''

continuity  = True
count = 0
while continuity == True and count < 20:
    count = count+1
    
    tokenized = word_tokenize(sample_seed)
    if len(tokenized) > maxlen:
        input_seed = tokenized[maxlen+1:]
        #print(input_seed)
    else:
        input_seed = tokenized
    
    sample_seed_vect = np.array([[word_indexes[c] if c in word_indexes.keys() else word_indexes['UNK'] \
                    for c in input_seed]])   
    sample_seed_vect = pad_sequences(sample_seed_vect, maxlen=maxlen)
    predicted = model.predict_classes(sample_seed_vect, verbose=0)
    for p in predicted:    
        #print(get_word_by_index(p, word_indexes))
        new_word = get_word_by_index(p, word_indexes)
        sample_seed = sample_seed+' '+new_word
        response_seed = response_seed+' '+new_word
response_seed
