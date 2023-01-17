import numpy as np # linear algebra

import os

from nltk import *

from nltk.tokenize import word_tokenize,wordpunct_tokenize

from string import punctuation

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

from keras.models import load_model

from nltk.corpus import stopwords

import h5py

filepath = '../input/'

filename = os.listdir(filepath)

print("Using these files : ",filename)

# Any results you write to the current directory are saved as output.
blog = ''

news = ''

blog += open(filepath+filename[1],'r',encoding='utf8').read()

news += open(filepath+filename[2],'r',encoding='utf8').read()
blog_tokens = wordpunct_tokenize(blog)

news_tokens = wordpunct_tokenize(news)
blog_tokens[:2],news_tokens[:2]
# List Tokenize word

tokenize_word_blog = []

tokenize_word_news = []

for word in blog_tokens:

    if len(word) >= 2 and word.isalpha() and word.lower()!='the':

        word = word.replace('?','')

        word = word.replace('.','')

        word = word.replace('!','')

        word = word.replace(';','')

        word = word.replace(':','')

        tokenize_word_blog.append(word.lower())



for word in news_tokens:

    if len(word) >= 2 and word.isalpha() and word.lower()!='the':

        word = word.replace('?','')

        word = word.replace('.','')

        word = word.replace('!','')

        word = word.replace(';','')

        word = word.replace(':','')

        tokenize_word_blog.append(word.lower())
final_tokenize_word = []

final_tokenize_word += tokenize_word_blog[:40000]

final_tokenize_word += tokenize_word_news[:40000]
final_tokenize_words = []

for i in final_tokenize_word:

    if i=='ve':

        final_tokenize_words.append('have')

    elif i=='re':

        final_tokenize_words.append('are')

    elif i=='ll':

        final_tokenize_words.append('will')

    else:

        final_tokenize_words.append(i)
# pickle.dump(final_tokenize_words,open('tokenized_words.pkl','wb'))
tokenizer = Tokenizer() # creating object of Tokenizer()

tokenizer.fit_on_texts([final_tokenize_words])

encoded = tokenizer.texts_to_sequences([final_tokenize_words])[0]
encoded[:5]
vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)
sequences = list()

for i in range(1, len(encoded)):

    sequences.append(encoded[i-1:i+1])

print('Total Sequences: %d' % len(sequences))
sequences[:5]
sequences = np.array(sequences) # Converting list to numpy array

X, Y = sequences[:,0],sequences[:,1]
Y = to_categorical(Y,num_classes=vocab_size)
model = Sequential()

model.add(Embedding(vocab_size, 400, input_length=1))

model.add(LSTM(400,return_sequences=True))

model.add(LSTM(400))

model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(X, Y, epochs=200,batch_size=48, verbose=2)
# serialize model to HDF5

model.save("new_model.h5")

print("Saved model to disk")
def generate_seq(word):

    in_text, result = word, word

    # generate a fixed number of words

    for _ in range(3):

        # encode the text as integer

        encode = tokenizer.texts_to_sequences([in_text])[0]

        encode = np.array(encode)

        # predict a word in the vocabulary

        yhat = model.predict_classes(encode, verbose=0)

        # map predicted word index to word

        out_word = ''

        for word, index in tokenizer.word_index.items():

            if index == yhat:

                out_word = word

                break

        # append to input

        in_text, result = out_word, result + ' ' + out_word

    print(result)
generate_seq('how')
generate_seq('so')
generate_seq('you')
generate_seq('what')
generate_seq('when')