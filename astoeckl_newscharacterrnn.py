import pandas as pd
import numpy as np

df = pd.read_csv("../input/news_en.csv", sep=',',index_col = "id")
print(df.shape)
df.head()
text = df["Headline"].str.cat(sep='\n')
text_size = len(text)
print('Text Size: %d' % text_size)
from pickle import dump

chars = sorted(list(set(text)))
mapping = dict((c, i) for i, c in enumerate(chars))
dump(mapping, open('mapping.pkl', 'wb'))

vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
encoded_text = [mapping[char] for char in text]
encode_size = len(encoded_text)
print('Code Size: %d' % encode_size)
seqlen = 10
batchsize = 512
batchnum = int((encode_size - seqlen) / batchsize)

from keras.utils import to_categorical

def myGenerator():
    while 1:
        for i in range(batchnum): 
            X_batch = []
            y_batch = []
            for j in range(batchsize):
                X_batch.append(encoded_text[i*batchsize+j:i*batchsize+j+seqlen])
                y_batch.append(encoded_text[i*batchsize+j+seqlen:i*batchsize+j+seqlen+1])
                
            X_batch = np.array([to_categorical(x, num_classes=vocab_size) for x in X_batch])
            y_batch = np.array(to_categorical(y_batch, num_classes=vocab_size))

            yield (X_batch, y_batch)
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, SimpleRNN
from keras.models import Model

model = Sequential()
model.add(LSTM(300, return_sequences=True, input_shape=(seqlen, vocab_size)))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(75))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
my_generator = myGenerator()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(my_generator, steps_per_epoch = batchnum, epochs = 50, verbose=1)
model.save('model_3lay_50.h5')
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import random
 
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char2] for char2 in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        encoded = to_categorical(encoded, num_classes=len(mapping))
        probs = model.predict_proba(encoded)
        yhat = random.choices(range(0,vocab_size), weights=probs[0], k=1)[0]
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        in_text += out_char
        if char =="\n":
            break
    return in_text
 
model = load_model('model_3lay_20.h5')
mapping = load(open('mapping.pkl', 'rb'))
 
print(generate_seq(model, mapping, seqlen, 'Tump tells', 400))
print(generate_seq(model, mapping, seqlen, 'Erdogan is', 400))
print(generate_seq(model, mapping, seqlen, 'Clinton is', 400))