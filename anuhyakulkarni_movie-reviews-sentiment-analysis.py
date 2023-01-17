#import all the essential libraries

import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords 
# To find out the name of the input file
import os

#import the csv as df
df= pd.read_csv('/kaggle/input/movie-review/movie_review.csv')

df.head()
#replace pos and neg as '1' and 'o'
df.tag[df.tag == 'pos'] = 1
df.tag[df.tag == 'neg'] = 0
# shuffle the df well and check for the tag column changes
from sklearn.utils import shuffle
df = shuffle(df)
df.head() 
import seaborn as sns
sns.countplot(x='tag',data=df)
df.tag
#set tag as target
target = df['tag'].values
target = np.array(target, dtype='int64')
# we use lemmatization techniques to reduce the number of words in the vocabulary
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer() 

def clean_word(word_list):
    global STOPWORDS
    new = []
    for word in word_list:
        word = word.replace('.', '')
        word = word.replace(',', '')
        word = word.replace(';', '')
        word = word.lower()
        if (word.isalpha() or word.isdigit()) and word not in STOPWORDS:   
            new.append(lemmatizer.lemmatize(word))
    return new
from sklearn.preprocessing import LabelEncoder

text = df['text'].values

# Tokenize each sentence 
text_arr = [row.split(' ') for row in text]
vocab = []
clean_text_array = []
for row in text_arr:
    clean_row = clean_word(row)
    clean_text_array.append(clean_row)
    vocab.extend(clean_row)
#set and list all the words to vocabulary and print the length(to get a count of the number of words)
vocabulary = list(set(vocab))
len(vocabulary)

vectorizer = LabelEncoder()
vectorizer.fit(vocabulary)

# Create token vector using Label Encoder fit on entire vocabulary
token_vector = []
# declare max_words to keep count of the longest sentence vectorized
# we need this to pad every other vector to same length as longest vector

max_words = 0 
for row in clean_text_array:
    encoded = vectorizer.transform(row).tolist()
    size = len(encoded)
    if size>max_words: 
        max_words=size
    token_vector.append(encoded)
max_words #print max_words
# pad each sentence with zeros to the length of the longest sentence
padded = []
for row in token_vector:
    r = np.pad(row, (0, max_words-len(row)), 'constant')
    padded.append(r)
# all padded sentences to example vector

ex_vector = np.array(padded)
# split train and test data into 80:20, data=ex_vector, target=tag

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(ex_vector,target, test_size=0.2)
import tensorflow as tf
train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
test_dataset = tf.data.Dataset.from_tensor_slices((xtest, ytest))
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
type(train_dataset)
from tensorflow import keras  
embedding_dim=16 

# defining the sequential model with an Embedding layer
# Add a Global Average Pooling 1D layer to flattent the matrix into vector

model = keras.models.Sequential([
  keras.layers.Embedding(33617, embedding_dim), #130590 as input based on vocabulary
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(32, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid')
])
#complie the model
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
epochs=5                  #five iterations
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=2)
# Plot Accuracy
plt.plot(range(epochs), history.history['accuracy'])
plt.plot(range(epochs), history.history['val_accuracy'])

# Plot Loss
plt.plot(range(epochs), history.history['loss'])
plt.plot(range(epochs), history.history['val_loss'])
pred = model.predict_classes(xtest)
cm = confusion_matrix(ytest,pred)
cm
plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['neg','pos'] , yticklabels = ['neg','pos'])