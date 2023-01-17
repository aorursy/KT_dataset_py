

from keras.models import Sequential, Model

from keras.layers import Dense, Activation, Dropout

from keras.layers import LSTM, Bidirectional

from keras.metrics import categorical_accuracy

import numpy as np

import os

import collections



#import spacy, and english model

import spacy

nlp = spacy.load("en_core_web_sm")
data_dir = '../input/'# data directory containing input.txt

save_dir = '' # directory to store models

seq_length = 10 # sequence length

sequences_step = 1 #step to create sequences
def create_wordlist(doc):

    wl = []

    ignore=('0','1','2','3','4','5','6','7','8','9','_','--')

    for word in doc:

        if word.text.isdigit()==False and word.text.startswith("\n")==False and word.text.startswith(ignore)==False:

            wl.append(word.text.lower())

            

    return wl
wordlist = []

input_file = os.path.join('../input/abcd.txt')

f= open(input_file,'r',errors='ignore',encoding='utf-8')

data = f.read()

doc=nlp(data)



wl = create_wordlist(doc)



wordlist = wordlist + wl



# count the number of words

word_counts = collections.Counter(wordlist)



# Mapping from index to word : that's the vocabulary

vocabulary_inv = [x[0] for x in word_counts.most_common()]

vocabulary_inv = list(sorted(vocabulary_inv))



# Mapping from word to index

vocab = {x: i for i, x in enumerate(vocabulary_inv)}#with index

words = [x[0] for x in word_counts.most_common()]



#size of the vocabulary

vocab_size = len(words)

print("vocab size: ", vocab_size)



#create sequences

sequences = []

next_words = []

for i in range(0, len(wordlist) - seq_length, sequences_step):

    

    sequences.append(wordlist[i: i + seq_length])

    next_words.append(wordlist[i + seq_length])



print('nb sequences:', len(sequences))


X_NPP = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y_NPP = np.zeros((len(sequences), vocab_size), dtype=np.bool)




for i, sentence in enumerate(sequences):

    for t, word in enumerate(sentence):

        X_NPP[i, t, vocab[word]] = 1

    y_NPP[i, vocab[next_words[i]]] = 1
rnn_size = 256 # size of RNN

batch_size = 64 # minibatch size

seq_length = 10 # sequence length

sequences_step = 1 #step to create sequences


def bidirectional_lstm_model(seq_length, vocab_size):

    print('Build LSTM model.')

    model_np = Sequential()

    model_np.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))

    model_np.add(Dropout(0.3))

    model_np.add(Dense(vocab_size))

    model_np.add(Activation('softmax'))

    model_np.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    

    return model_np
md = bidirectional_lstm_model(seq_length, vocab_size)

md.summary()
history = md.fit(X_NPP, y_NPP,

                 batch_size=64,

                 epochs=20,

                 validation_split=0.1

                )
#save_dir


score,accuracy=md.evaluate(X_NPP,y_NPP)

print("score",score)

print("accuracy: ",accuracy)
#save the model

md.save('my_model_gen_sentences_lstm.final.h5')
from keras.models import load_model

# load the model

print("loading model...")

model = load_model('my_model_gen_sentences_lstm.final.h5')
def sample(preds, temperature=1.0):

    # helper function to sample an index from a probability array

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)
 #initiate sentences

seq_length=10

seed_sentences = "how"

seed_sentences=seed_sentences.lower()

generated = ''

sentence = []

for i in range (seq_length):

        sentence.append(".")

    

seed = seed_sentences.split()





for i in range(len(seed)):

    sentence[seq_length-i-1]=seed[len(seed)-i-1]



generated += ' '.join(sentence)

start=len(generated)-len(seed_sentences)

print('Generating text with the following seed: "' + ' '.join(sentence) + '"')



words_number = 10

    #generate the text

for i in range(words_number):

        #create the vector

    x = np.zeros((1, seq_length, vocab_size))

       

    for t, word in enumerate(sentence):

        x[0, t, vocab[word]] = 1.

        



        #calculate next word

    preds = md.predict(x, verbose=0)[0]

  

    next_index = sample(preds, 0.34)

    next_word = vocabulary_inv[next_index]



        #add the next word to the text

    generated += " " + next_word



        # shift the sentence by one, and and the next word at its end

    sentence = sentence[1:] + [next_word]



print(generated[start:])








