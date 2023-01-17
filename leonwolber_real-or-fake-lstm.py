import pandas as pd
import numpy as np
import os
import re
import string

import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

import matplotlib.style as style
style.use('fivethirtyeight')

os.chdir("/home/leon/Documents/projects")
true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
true.head()
fake.head()
fake.iloc[0]['text']
true['label']  = pd.Series([0] * len(true)) # add label column | 0 == True, 1 == Fake
fake['label']  = pd.Series([1] * len(fake))

true['label'] = pd.Categorical(true['label']) # make label categorical
fake['label'] = pd.Categorical(fake['label'])
full = true.append(fake) # create single df
full
def remove_line_breaks(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text

#remove punctuation
def remove_punctuation(text):
    re_replacements = re.compile("__[A-Z]+__")  # such as __NAME__, __LINK__
    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
    '''Escape all the characters in pattern except ASCII letters and numbers: word_tokenize('ebrahim^hazrati')'''
    tokens = word_tokenize(text)
    tokens_zero_punctuation = []
    for token in tokens:
        if not re_replacements.match(token):
            token = re_punctuation.sub(" ", token)
        tokens_zero_punctuation.append(token)
    return ' '.join(tokens_zero_punctuation)

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def lowercase(text):
    text_low = [token.lower() for token in word_tokenize(text)]
    return ' '.join(text_low)

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in word_tokens if word not in stop])
    return text


#remove one character words
def remove_one_character_words(text):
    '''Remove words from dataset that contain only 1 character'''
    text_high_use = [token for token in word_tokenize(text) if len(token)>1]      
    return ' '.join(text_high_use)   
    
#%%
# Stemming with 'Snowball stemmer" package
def stem(text):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    text_stemmed = [stemmer.stem(token) for token in word_tokenize(text)]        
    return ' '.join(text_stemmed)

def remove_numbers(text):
    no_nums = re.sub(r'\d+', '', text)
    return ''.join(no_nums)


def full_clean(text):
    _steps = [
    remove_line_breaks,
    remove_one_character_words,
    remove_special_characters,
    lowercase,
    remove_punctuation,
    remove_stopwords,
    stem,
    remove_numbers
]
    for step in _steps:
        text=step(text)
    return text   
#%%
full['clean'] = [full_clean(i) for i in full['text']]
full.to_csv('fake_real.csv')
full = pd.read_csv("../input/fakereal/fake_real.csv")
articles = full['clean'].dropna().to_list()
articles[:2]
plt.figure(figsize=(16,13))
wc = WordCloud(background_color="black", max_words=1000, max_font_size= 200,  width=1600, height=800)
wc.generate(" ".join(articles))
plt.title("Most discussed terms", fontsize=20)
plt.imshow(wc.recolor( colormap= 'viridis' , random_state=17), alpha=0.98, interpolation="bilinear", )
plt.axis('off')
import tensorflow as tf

from tensorflow import keras

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras import optimizers


from collections import Counter
from sklearn.model_selection import train_test_split
def balanced_subsample(y, size=None):
    
    '''Sample from data and keep the classes balanced'''

    subsample = []

    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()

    return subsample
full.head()
# sample 10% of the data
sample = balanced_subsample(full['label'], len(full)*0.1) 

# extract the indices picked by the function to create subsample
sample = full.iloc[sample, :]  
text = sample['clean'].astype(str)
docs = text.to_list()

labels = pd.array(sample['label'])
print(len(docs))
# print('\n')
print(len(labels))
# check number of unique words to estimate a reasonable vocab size

one_str = ''.join(text)
unique_words = Counter(one_str.split())
len(unique_words) 
vocab_size = 46856  # 20000 for the entire dataset
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs[:1]) # list of lists
x = []

for i in encoded_docs:
    x.append(len(i))
    
print("In my sample the largest document has", max(x), "words.")
# now every document will be represented by a vector of the same length: 3000 values / 'words'

max_length = 2996
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


X = padded_docs
y = sample['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Embedding(input_dim = vocab_size, output_dim = 32, input_length = max_length))

model.add(Bidirectional(LSTM(64, activation='linear')))
model.add(Dense(32, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
   
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
   
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
# fit the model
history = model.fit(X_train, y_train, epochs=5, verbose=1, batch_size=30, validation_split = 0.2)
# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()