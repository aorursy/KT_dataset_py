## download the nltk

import nltk

nltk.download('all-corpora')
from nltk.tokenize      import sent_tokenize,word_tokenize

from nltk.corpus        import stopwords

from nltk.tokenize      import word_tokenize
### this is fetching the stop words

example_sentences  = "this is an example of the stop words"

stop_words         = set(stopwords.words("english"))

print(stop_words)
## this is the word tokenization

words = word_tokenize(example_sentences);

print(words)

filtered_sentence = []

for w in words:

  if w not in stop_words:

    filtered_sentence.append(w)

print(filtered_sentence)
## this is the stemming [we fix the world like listen and listening and make it one so it will use one only]

from nltk.stem import PorterStemmer

ps      = PorterStemmer()

example_word = ['listen','listenning','listened']

for w in example_word:

  print(ps.stem(w))
## stemming  a new word

new_text  = "it is very important to be pythonly when solve the problem in python. be pythonic. all pythoner do this job solved the problem in  a pythonic way"

words      =  word_tokenize(new_text)

stemmed_word = []

for w in words:

  stemmed_word.append(ps.stem(w))

print(stemmed_word)
from nltk.corpus    import state_union

from nltk.tokenize  import PunktSentenceTokenizer              ## unsupervised tokenizer

train_text  = state_union.raw('2005-GWBush.txt')

sample_text = state_union.raw("2006-GWBush.txt")

customn_sent_tokenizer = PunktSentenceTokenizer(train_text)   ## trian

print(customn_sent_tokenizer)

tokenize = customn_sent_tokenizer.tokenize(sample_text)       ## inference

print(tokenize)                                               ## this is tokenize win the sentence
## this is parts of speech tagging 

## find the parts of speech from the word coupous

tokenized_word = []

try:

  for i in tokenize:

    words  = nltk.word_tokenize(i)

    ## indexing the word with parts of speetch

    tagged = nltk.pos_tag(words)

    tokenized_word.append(tagged)

except:

  pass

print("Word and the parts of sppech")

print(tokenized_word)
## movie Reviwe for POSITIVE OR NEGATIVE WITH NLTK

import nltk

import random

from nltk.corpus import movie_reviews
documents = []

for category in movie_reviews.categories():

  for movie in movie_reviews.fileids(category):

    documents.append((list(movie_reviews.words(movie)),category))
print(documents[0])
random.shuffle(documents)
## find the most common words

all_words = []

for w in movie_reviews.words():

  all_words.append(w.lower())

## find the frequency of any word

## which word is the most used words

all_words = nltk.FreqDist(all_words)

print(all_words['disgusting'])

print(all_words['excellent'])
## take the feture

## we take the most used 4000 words as a feture

word_feature = list(all_words)[:3000]

print(word_feature)
## check if any word of he 3000 words in  the document

def find_features(document):

  words    = set(document)

  features = {}

  for w in word_feature:

    features[w] = (w in words)



  return features 
print(find_features(movie_reviews.words('neg/cv000_29416.txt')))
feature_set = []

for x,y in documents:

  feature_set.append((find_features(x),y))
feature_set[:1]
train_set = feature_set[:1900]

test_set = feature_set[1900:]
classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.accuracy(classifier,test_set)

print("accuracy : {}".format(accuracy))
classifier.show_most_informative_features(40)
### now we are make a recomendation system based on the sentiment 

### with neural network
import numpy             as np

import pandas            as pd

import seaborn           as sns

import matplotlib.pyplot as plt
## import the amazon review dataset

df = pd.read_csv("../input/amazon-dataset-for-recomendation/1429_1.csv")
df.head()
## make a plotting of how many review and how many rating
review =  pd.DataFrame(df.groupby('reviews.rating').size().sort_values(ascending=True).rename("Users").reset_index())

review.plot(kind = "bar")
### so we have 4 rating the most

### now make a prediction system

df = df[['reviews.rating' , 'reviews.text']]

df.head()
df.isnull().sum()
df = df.dropna()

df.head()
review = list(df['reviews.text'])     ## feature

rating = list(df['reviews.rating'])   ## target 
print(review.__len__())

print(rating.__len__())
size = review.__len__()
training_size = 30000

training_sentences = review[0:training_size]

testing_sentences = review[training_size:]

training_labels = rating[0:training_size]

testing_labels = rating[training_size:]
## we need to tokenize the word

## we use tensorlfow tokenizer this time

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 30000

oov_token  ="<OOV>"

tokenizer  = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
print(word_index)
## now make the sequence based on the bag of words

training_sequences = tokenizer.texts_to_sequences(training_sentences)
## apply padding

from tensorflow.keras.preprocessing.sequence import pad_sequences

print(training_sequences[0])

print(training_sequences[1])
training_padded = pad_sequences(training_sequences,maxlen=32,padding="post",truncating="post")
print(training_padded[0].__len__())

print(training_padded[1].__len__())
print(training_padded.shape)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences) 

testing_padded = pad_sequences(testing_sequences,maxlen=32,padding="post",truncating="post")
print(testing_padded[0].__len__())

print(testing_padded[1].__len__())
print(testing_padded.shape)
## this is the Embedding Layer Parameter

vocab_size = 20000

embedding_dim = 8

max_length = 32
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length))

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(6,activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.summary()
training_labels = np.array(training_labels)

testing_labels = np.array(testing_labels)
num_epochs=3
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels))
model.evaluate(testing_padded,testing_labels)
predicted = model.predict(testing_padded)
actual=[]

for item in predicted:

    actual.append(np.argmax(item))
## imdb dataset review

import numpy             as np

import pandas            as pd

import seaborn           as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.head()
d = {"positive":1,"negative":0}
df['sentiment'] = df['sentiment'].map(d)
df.head()
review =  pd.DataFrame(df.groupby('sentiment').size().sort_values(ascending=True).rename("Users").reset_index())

review.plot(kind="bar")
df.__len__()
review = list(df['review'])     ## feature

rating = list(df['sentiment'])   ## target 
training_size = 30000

training_sentences = review[0:training_size]

testing_sentences = review[training_size:]

training_labels = rating[0:training_size]

testing_labels = rating[training_size:]
## we need to tokenize the word

## we use tensorlfow tokenizer this time

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 40000

oov_token  ="<OOV>"

tokenizer  = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

print(word_index)
training_sequences = tokenizer.texts_to_sequences(training_sentences)
## apply padding

from tensorflow.keras.preprocessing.sequence import pad_sequences

training_padded = pad_sequences(training_sequences,maxlen=250,padding="post",truncating="post")
print(training_padded[0].__len__())

print(training_padded[1].__len__())
testing_sequences = tokenizer.texts_to_sequences(testing_sentences) 

testing_padded = pad_sequences(testing_sequences,maxlen=250,padding="post",truncating="post")
## this is the Embedding Layer Parameter

## this vocab size must be greater than the tokenizer vocab size

vocab_size = 50000

embedding_dim = 100

max_length = 33
training_labels = np.array(training_labels)

testing_labels = np.array(testing_labels)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length))

model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(250,activation='relu'))

model.add(tf.keras.layers.Dropout(.2))

model.add(tf.keras.layers.Dense(2,activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
num_epochs=3
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels))