import numpy as np
import pandas as pd 
import re
import string
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
def load_docs(filename):
    file = open(filename,'r',encoding = 'utf-8-sig')
    text = file.read()
    file.close()
    return(text)

def clean_docs(doc):
    re_punc = re.compile('[%s]' % re.escape("\n"))
    doc = doc.replace('--', ' ')
    tokens = re_punc.sub(' ',doc) 
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return(tokens)

def generate_pred(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded],maxlen = seq_length, truncating='pre')
        yhat_f = model.predict(encoded,verbose = 0)
        #yhat = np.argmax(yhat_f) # Not use the word with max prob everytime...
        yhat = np.random.choice(len(yhat_f[0]), p=yhat_f[0]) # but sample with distributionn yhat like above
        out_word = ''
        for word,index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text = ' ' + out_word
        result.append(out_word)
    return(' '.join(result))
# Create list of words
a = load_docs("../input/plato-republic/the_republic_clean.txt")
tokens = clean_docs(a)

# Organize into sequences of tokens
length = 50 + 1
my_sequences = list()
for i in range(length, len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    my_sequences.append(line)

# Tokenize the sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(my_sequences)
sequences = tokenizer.texts_to_sequences(my_sequences)
sequences = np.array(sequences)

X,y = sequences[:,:-1],sequences[:,-1]
vocab_size = len(tokenizer.word_index) + 1
seq_length = X.shape[1]
y = to_categorical(y, num_classes=vocab_size)
X.shape
# Define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=seq_length))
model.add(LSTM(100, return_sequences=True, recurrent_dropout=0.2))
model.add(LSTM(100, recurrent_dropout=0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model
model.fit(X, y, batch_size=128, epochs=100)

model.summary()
#model.save('model_lstm_1.h5')
# load the model
model2 = load_model('../input/plato-lstm-1/model_lstm_1.h5')
from random import randint
lines = a.split('\n')
seed_text = lines[randint(0,len(lines))]
print(seed_text)
print("\n")
print(generate_pred(model2, tokenizer, seq_length, seed_text, 50))
# Improve: add more data (5 times as much), tune LSTM layer, 