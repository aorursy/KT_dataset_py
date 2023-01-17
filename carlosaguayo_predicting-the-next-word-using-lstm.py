plato = open("../input/the-republic-by-plato/1497.txt", "r").read()
print (plato[:53])

# http://www.gutenberg.org/cache/epub/730/pg730.txt
oliver_twist = open("../input/dickens/dickens/pg730.txt", "r").read()
print (oliver_twist[:64])

christmas_carol = open("../input/dickens/dickens/pg19337.txt", "r").read()
print (christmas_carol[:69])

text = oliver_twist
import string

# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    tokens = [' ' if w in string.punctuation else w for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens
 
tokens = clean_doc(text)

number_of_unique_tokens = len(set(tokens))

print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % number_of_unique_tokens)
print('These are the first 200 tokens: %s' % tokens[:200])

# A key design decision is how long the input sequences should be. 
# They need to be long enough to allow the model to learn the context for the words to predict. 
# This input length will also define the length of seed text used to generate new sequences 
# when we use the model.
# There is no correct answer. With enough time and resources, we could explore the ability of 
# the model to learn with differently sized input sequences.

sequence_length = 2

# organize into sequences of tokens of input words plus one output word
length = sequence_length + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)

print ('Total Sequences: %d' % len(sequences))
print ('This is the first sequence: {0}'.format(sequences[0]))
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)
# vocab_size = len(tokenizer.word_index) + 1
vocab_size = number_of_unique_tokens + 1
 
sequences0 = np.array(sequences)
X, y = sequences0[:,:-1], sequences0[:,-1]
y = to_categorical(y, num_classes=vocab_size)

# The learned embedding needs to know how many dimensions will be used to represent each word. 
# That is, the size of the embedding vector space. That is, the size of the embedding vector space.
# Common values are 50, 100, and 300. Consider testing smaller or larger values.
dimensions_to_represent_word = 100
 
model = Sequential()
model.add(Embedding(vocab_size, sequence_length, input_length=sequence_length))
# We will use a two LSTM hidden layers with 100 memory cells each. 
# More memory cells and a deeper network may achieve better results.
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training may take a few hours on modern hardware without GPUs. 
# You can speed it up with a larger batch size and/or fewer training epochs.
# model.fit(X, y, batch_size=128, epochs=100)
model.fit(X, y, batch_size=128, epochs=100)
print (X.shape)
prediction = model.predict(X[0].reshape(1,sequence_length))
print (prediction.shape)
print (prediction)
test = ['thank you',
'welcome to',
'when there',
'more than',
'it cannot',
'is that',
'although this',
'do you',
'I was',
'the only',
'a great']

for t in test:
    example = tokenizer.texts_to_sequences([t])
    prediction = model.predict(np.array(example))
    predicted_word = np.argmax(prediction)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))  # https://stackoverflow.com/a/43927939/246508
    print ("{0} -> {1}".format(t, reverse_word_map[predicted_word]))