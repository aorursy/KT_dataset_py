import pandas as pd
import numpy as np
import string
import re
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
print(train.shape, test.shape)
pres = {'deKlerk': 0,
        'Mandela': 1,
        'Mbeki': 2,
        'Motlanthe': 3,
        'Zuma': 4,
        'Ramaphosa': 5}

train.replace({'president': pres}, inplace=True)
# speech number: intro lines
starts = {
    0: 1,
    1: 1,
    2: 1,
    3: 12,
    4: 12,
    5: 5,
    6: 1,
    7: 1,
    8: 8,
    9: 9,
    10: 12,
    11: 14,
    12: 14,
    13: 15,
    14: 15,
    15: 15,
    16: 15,
    17: 15,
    18: 15,
    19: 15,
    20: 20,
    21: 1,
    22: 15,
    23: 20,
    24: 20,
    25: 15,
    26: 15,
    27: 20,
    28: 20,
    29: 15,
    30: 18
}
def divide_on(df, char):
    
    # iterate over text column of DataFrame, splitting at each occurrence of char

    sentences = []
    # let's split the data into senteces
    for i, row in df.iterrows():
        
        # skip the intro lines of the speech
        for sentence in row['text'].split(char)[starts[i]:]:
            sentences.append([row['president'], sentence])

    df = pd.DataFrame(sentences, columns=['president', 'text'])
    
    return df[df['text'] != '']
train = divide_on(train, '.')
train.head(5)
train.shape
train['president'].value_counts()
# proportion of total
train['president'].value_counts()/train.shape[0]
train['sentence'] = None
test['president'] = None

df = pd.concat([train, test], axis=0, sort=False)
# reorder columns
df = df[['sentence', 'text', 'president']]
df.tail()
def fixup(text):
    
    # remove punctuation
    text = ''.join([char for char in text if char == '-' or char not in string.punctuation])
    # remove special characters
    text = text.replace(r'^[*-]', '')
    # lowercase
    text = text.lower()
    
    # remove hanging whitespace
    text = " ".join(text.split())
    
    return text


df['text'] = df['text'].apply(fixup)
df.head(5)
# get length of sentence as variable
df['length'] = df['text'].apply(len)
# what are our longest sentences?
df.sort_values(by='length', ascending=False).head(10)
df.loc[3930][1]
# what are our shortest sentences?
df.sort_values(by='length').head(5)
# let's check the shortest sentences in our test set
df[pd.isnull(df['president'])].sort_values(by='length').head()
# sentences with just a few characters are of no use to us
df = df[df['length']>10]
# what are our shortest sentences now?
df.sort_values(by='length').head(5)
df['president'].value_counts()
from collections import Counter
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Activation, Bidirectional, RepeatVector, Input
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
train = df[pd.isnull(df['sentence'])]
test = df[pd.notnull(df['sentence'])]
# load whole corpus of words
all_text = df['text'].values
word_counts = {}

# iterate over each sentence in the corpus
for text in all_text:
    
    for word in text.split():
    
        # if the word isn't in the dictionary
        if word not in word_counts.keys():
          # create a new index for it, and add 1 to its count
          word_counts[word] = 1

        # if word is already in the dictionary
        else:
          # find its key, and add 1 to its count
          word_counts[word] += 1

# create mappings
stoi = {}  # string to index
itos = {}  # index to string

idx = 1

# order the words from most to least common
for word in sorted(word_counts, key=word_counts.get, reverse=True):

    if word not in stoi.keys():
        stoi[word] = idx
        itos[idx] = word
        idx += 1
sorted(word_counts, key=word_counts.get, reverse=True)[:25]
# how many words in the corpus?
vocabulary_size = len(stoi.keys())
print(vocabulary_size)
def tokenize(string):

    ''' convert string to tokens '''
    
    sequence = []
    
    for word in string.split():
        sequence.append(stoi[word])
    
    return sequence
def untokenize(tokens):
    
    ''' convert tokens to string '''
    
    string = ' '.join(itos[token] for token in tokens)
    
    return string
# view the original text data
train['text'][600]
# tokenize the training data
train['tokens'] = train['text'].apply(tokenize)

# tokenize the test data
test['tokens'] = test['text'].apply(tokenize)
# show the tokenized data
train['tokens'][600]
# convert back to text
untokenize(train['tokens'][600])
# save the tokenized data as features for training
X_train = train['tokens'].values
X_test = test['tokens'].values
def one_hot_encode(label):
    
    # initialize zero array
    vec = [0, 0, 0, 0, 0, 0]
    
    # set index of array corresponding to label = 1
    vec[label] = 1
    
    return vec

# save encoded labels as target for model
y_train = np.vstack(row for row in train['president'].apply(one_hot_encode).values)
y_train[600]
print('Train size:', X_train.shape)
print('Test size:', X_test.shape)
X_train
# show a single observation
print('--token sequence')
print(X_train[325])
print('--words--')
print(untokenize(X_train[325]))
print('--label--')
print(y_train[325])
print('Maximum train review length: {}'.format(
len(max(X_train, key=len))))

print('Maximum test review length: {}'.format(
len(max(X_test, key=len))))
untokenize(max(X_train, key=len))
untokenize(max(X_test, key=len))
max_words = len(max(X_train, key=len))
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
# example sentence
X_test[1]
embedding_size=32

model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(Dropout(0.05))
# model.add(Conv1D(64, 5, activation="relu"))
# model.add(MaxPooling1D(pool_size=4))
model.add(Bidirectional(LSTM(64, activation='tanh')))
model.add(Dropout(0.05))
model.add(Dense(6, activation='softmax'))

print(model.summary())
batch_size = 128
num_epochs = 15

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.25)
predictions = model.predict(X_test)
pred_lbls = []
for pred in predictions:
    pred = list(pred)
    max_value = max(pred)
    max_index = pred.index(max_value)
    pred_lbls.append(max_index)

predictions = np.array(pred_lbls)
test['president'] = predictions
test['president'].value_counts()
train['president'].value_counts()
test[['sentence', 'president']].to_csv('rnn_1.csv', index=False)
