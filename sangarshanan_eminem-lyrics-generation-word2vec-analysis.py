import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
import plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go
data = pd.read_csv("../input/songdata.csv")
data.head(5)
del data['link']
len(list(data['artist'].unique()))
eminem = data[(data['artist'] == 'Eminem')]
eminem.head(3)
rap = list(eminem['text'])
word=[]
for i in range(0,len(rap)):
    kk = rap[i].replace("\n"," ")
    s=kk.split(' ')
    o = [x for x in s if x]
    word.append(o)
word = [j for i in word for j in i]
word[:5]
el = ["i'm","get","got"]
stop = set(stopwords.words('english'))
word = [word.lower() for word in word]
words = [i for i in word if i not in stop]
words = [i for i in words if i not in el]
for i in range(0,len(words)):
    words[i] = re.sub(r'[^\w\s]','',words[i])
words = [x for x in words if x]
from collections import Counter
labels, values = zip(*Counter(words).items())
w = Counter(words)
s = w.most_common(20)
x , y = zip(*(s))
data = [go.Bar(x=x,y=y)]
layout = go.Layout(
    title='Words in Eminem Lyrics ',
    xaxis=dict(
        title='Words Used',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Number of times it was used',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
iplot(go.Figure(data=data, layout = layout))
unique_words = sorted(set(words))
len(unique_words)
from keras.models import Sequential
from keras.layers.noise import GaussianNoise
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
rap = np.array(rap)
lyric = (''.join(rap))
l = set(lyric)
len(l)
vocab= [k for k in l] 
char_ix={c:i for i,c in enumerate(vocab)}
ix_char={i:c for i,c in enumerate(vocab)}
ix_char
maxlen=40
vocab_size=len(vocab)
sentences=[]
next_char=[]
for i in range(len(lyric)-maxlen-1):
    sentences.append(lyric[i:i+maxlen])
    next_char.append(lyric[i+maxlen])
sentences
X=np.zeros((len(sentences),maxlen,vocab_size))
y=np.zeros((len(sentences),vocab_size))
for ix in range(len(sentences)):
    y[ix,char_ix[next_char[ix]]]=1
    for iy in range(maxlen):
        X[ix,iy,char_ix[sentences[ix][iy]]]=1

from keras.layers import Activation,LSTM,Dense
from keras.optimizers import Adam
model=Sequential()
model.add(LSTM(128,input_shape=(maxlen,vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy')
model.fit(X,y,epochs=5,batch_size=128)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
import random
generated=''
start_index=random.randint(0,len(lyric)-maxlen-1)
sent=lyric[start_index:start_index+maxlen]
generated+=sent
for i in range(1900):
    x_sample=generated[i:i+maxlen]
    x=np.zeros((1,maxlen,vocab_size))
    for j in range(maxlen):
        x[0,j,char_ix[x_sample[j]]]=1
    probs=model.predict(x)
    probs=np.reshape(probs,probs.shape[1])
    ix=np.random.choice(range(vocab_size),p=probs.ravel())
    generated+=ix_char[ix]
generated.split("\n")
from unidecode import unidecode
def get_tokenized_lines(df):
    words = []
    
    for index, row in df['text'].iteritems():
        row = str(row).lower()
        for line in row.split('\n'):
            new_words = re.findall(r"\b[a-z']+\b", unidecode(line))
            words = words + new_words
        
    return words
all_lyric_lines = get_tokenized_lines(eminem)
SEQ_LENGTH = 50 + 1
sequences = list()

for i in range(SEQ_LENGTH, len(all_lyric_lines)):
    seq = all_lyric_lines[i - SEQ_LENGTH: i]
    sequences.append(seq)

print('Total Sequences: %d' % len(sequences))
vocab = set(all_lyric_lines)

word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for w, i in word_to_index.items()}
word_indices = [word_to_index[word] for word in vocab]
vocab_size = len(vocab)

print('vocabulary size: {}'.format(vocab_size))
def get_tokenized_lines(lines, seq_len):
    tokenized = np.zeros((len(lines), seq_len))
    
    for r, line in enumerate(lines):
        for c, word in enumerate(line):
            tokenized[r, c] = word_to_index[word]

    return tokenized

tokenized_seq = get_tokenized_lines(sequences, SEQ_LENGTH)
tokenized_seq[:, -1].shape
X, y = tokenized_seq[:, :-1], tokenized_seq[:, -1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = len(X[0])

print("X_shape", X.shape)
print("y_shape", y.shape)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=10)

seed_text = "Hailie i know you miss your mom and i know you miss your dad well i'm gone but i'm trying to give you the life that i never had i can see you're sad even when you smile even when you laugh i can see it in your eyes deep inside"
def texts_to_sequences(texts, word_to_index):
    indices = np.zeros((1, len(texts)), dtype=int)
    
    for i, text in enumerate(texts):
        indices[:, i] = word_to_index[text]
        
    return indices

def my_pad_sequences(seq, maxlen):
    start = seq.shape[1] - maxlen
    
    return seq[:, start: start + maxlen]

def generate_seq(model, word_to_index, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text

    for _ in range(n_words):
        encoded = texts_to_sequences(in_text.split()[1:], word_to_index)
        encoded = my_pad_sequences(encoded, maxlen=seq_length)
        
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
    
        for word, index in word_to_index.items():
            if index == yhat:
                out_word = word
                break
        
        in_text += ' ' + out_word
        result.append(out_word)
        
    return ' '.join(result)

generated = generate_seq(model, word_to_index, seq_length, seed_text, 50)
print(generated)

import gensim 
for i in range(0,len(word)):
        word[i] = [word.lower() for word in word[i]]
len(word)
model = gensim.models.Word2Vec(
        word,
        size=150,
        window=10,
        min_count=2,
        workers=10)
model.train(word, total_examples=len(word), epochs=10)

print(model.similarity('eminem', 'rap'))
print(model.similarity('eminem', 'marshall'))
model.most_similar('eminem')