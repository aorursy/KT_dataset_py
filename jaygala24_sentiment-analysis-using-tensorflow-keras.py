import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
input_path = '/kaggle/input/nlp-getting-started/'

train = pd.read_csv(os.path.join(input_path, 'train.csv'))
test = pd.read_csv(os.path.join(input_path, 'test.csv'))
train.head()
print('Train: ', train.shape)
print('Test: ', test.shape)
# check the missing values for keyword and location
len(train['keyword'].isnull()), len(train['location'].isnull())
# non disaster tweet
train[train['target'] == 0]['text'].values[0]
# disaster tweet
train[train['target'] == 1]['text'].values[0]
import re
import unicodedata
import spacy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam
# reference:~ https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/bonus%20content/nlp%20proven%20approach/contractions.py

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
# loading the spacy's en_core_web_sm
nlp = spacy.load('en_core_web_sm')
nlp.pipe_names
# create and add sentencizer to the pipeline
sent = nlp.create_pipe('sentencizer')
nlp.add_pipe(sent, before='parser')
nlp.pipe_names
def text_cleaning(text):
    """
    Returns cleaned text (Accented Characters, Expand Contractions, Special Characters)
    Parameters
    ----------
    text -> String
    """
    # remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # remove emails
    text = ' '.join([i for i in text.split() if '@' not in i])
    
    # remove urls
    text = re.sub('http[s]?://\S+', '', text)
    
    # expand contractions
    for word in text.split():
        if word.lower() in CONTRACTION_MAP:
            text = text.replace(word[1:], CONTRACTION_MAP[word.lower()][1:])
    
    # remove special characters
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    
    # remove extra white spaces
    text = re.sub('\s+', ' ', text)
    
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        if token.lemma_ != '-PRON-':
            tokens.append(token.lemma_.lower().strip())
        else:
            tokens.append(token.lower_)

    return ' '.join(tokens)
text_cleaning("I don't like this movie. The plot is   terrible :(")
# split the data into inputs and outputs
X_train = train['text'].apply(text_cleaning)
y_train = train['target']
X_test = test['text'].apply(text_cleaning)
X_train.head()
# split the train set into train and valid set
split = 0.8
train_size = int(len(X_train) * 0.8)
idx = np.random.permutation(X_train.index)
train_idx = idx[:train_size]
valid_idx = idx[train_size:]
train_data = X_train.iloc[train_idx]
train_labels = y_train.iloc[train_idx]
valid_data = X_train.iloc[valid_idx]
valid_labels = y_train.iloc[valid_idx]
oov_token = '<unk>'
padding_type = 'post'
trunc_type = 'post'
embedding_dim = 100
max_len = max([len(x) for x in train_data])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index
vocab_size = len(word_index)
print('Vocab size : ', vocab_size)
train_seq = tokenizer.texts_to_sequences(train_data)
train_pad = pad_sequences(train_seq, padding=padding_type, truncating=trunc_type, maxlen=max_len)

valid_seq = tokenizer.texts_to_sequences(valid_data)
valid_pad = pad_sequences(valid_seq, padding=padding_type, truncating=trunc_type, maxlen=max_len)
test_seq = tokenizer.texts_to_sequences(X_test)
test_pad = pad_sequences(test_seq, padding=padding_type, truncating=trunc_type, maxlen=max_len)
embeddings_idx = {}
glove_path = '/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt'
with open(glove_path, 'r') as f:
    for line in f:
        data = line.split()
        word = data[0]
        values = np.asarray(data[1:], dtype=np.float32)
        embeddings_idx[word] = values

print('Building Embedding Matrix...')
embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vec = embeddings_idx.get(word)
    if embedding_vec is not None:
        embeddings_matrix[i] = embedding_vec
print('Embedding Matrix Generating...')
print('Embedding Matrix Shape -> ', embeddings_matrix.shape)
# build a model
model =  keras.models.Sequential([
    Embedding(vocab_size + 1, embedding_dim, input_length=max_len, weights=[embeddings_matrix], trainable=False),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()
optimizer = Adam(lr=3e-4)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_pad, train_labels, epochs=20, validation_data=(valid_pad, valid_labels))
# model analysis
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('# epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
# load the sample submission csv file
sample_submission = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'))
# predict on test data
sample_submission['target'] = model.predict_classes(test_pad)
# save the sample submission csv file
sample_submission.to_csv('submission.csv', index=False)
