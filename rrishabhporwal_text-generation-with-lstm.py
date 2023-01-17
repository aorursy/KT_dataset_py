import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def read_files(filepath):

    with open(filepath) as f:

        str_txt = f.read()

        

    return str_txt
read_files('../input/moby_dick_four_chapters.txt')
import spacy

nlp = spacy.load('en', disable=['parser','tagger','ner'])

nlp.max_length = 1198623
def separate_punc(doc_text):

    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\ \n \n\\n \\n\\n\\n!\"-#$%&()--.*+,-/:;<=>?@[\\\\]^_`{|}~\\t\\n ']
d = read_files('../input/moby_dick_four_chapters.txt')

tokens = separate_punc(d)

len(tokens)
train_len = 25+1

text_sequences = []



for i in range(train_len,len(tokens)):

    seq = tokens[i-train_len:i]

    text_sequences.append(seq)
' '.join(text_sequences[0])
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_sequences)

sequences = tokenizer.texts_to_sequences(text_sequences)
for i in sequences[0]:

    print(f"{i} : {tokenizer.index_word[i]}")
tokenizer.word_counts
vocabulary_size = len(tokenizer.word_counts)

vocabulary_size
type(sequences)
import numpy as np
sequences = np.array(sequences)

sequences
from keras.utils import to_categorical
X = sequences[:,:-1]

y = sequences[:,-1]

y
y = to_categorical(y,num_classes=vocabulary_size+1)

y
seq_len = X.shape[1]
seq_len
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM
def create_model(vocabulary_size,seq_len):

    model = Sequential()

    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))

    model.add(LSTM(150, return_sequences=True))

    model.add(LSTM(150))

    model.add(Dense(vocabulary_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    model.summary()

    

    return model
model = create_model(vocabulary_size+1, seq_len)
from pickle import dump, load
model.fit(X, y, batch_size=150, epochs=200, verbose=1)
model.save('my_mobydick_model.h5')

dump(tokenizer, open('my_simpletokenizer','wb'))
from keras.preprocessing.sequence import pad_sequences
def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):

    output_text = [] 

    input_text = seed_text

    

    for i in range(num_gen_words):

        encoded_text = tokenizer.texts_to_sequences([input_text])[0]

        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')

        pre_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]

        pred_word = tokenizer.index_word[pre_word_ind]

        input_text += ' '+pred_word

        output_text.append(pred_word)

        

    return ' '.join(output_text) 
text_sequences[0]
import random

random.seed(101)

random_pick = random.randint(0, len(text_sequences))
random_seed_text = text_sequences[random_pick]
random_seed_text
seed_text = ' '.join(random_seed_text)
seed_text
generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=25)