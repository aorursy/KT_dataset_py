import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import string

import re

import nltk

import warnings

warnings.filterwarnings('ignore')



dataset = pd.read_csv('../input/fraud-email-dataset/fraud_email_.csv')

dataset.head()
print ("Number of Columns = ", dataset.shape[1])

print ("Number of rows = ", dataset.shape[0])
requiredColumns = dataset.columns.values

for x in requiredColumns:

    if(dataset[x].isnull().sum() > 0):

        print (x)

    
from nltk.corpus import stopwords

import string



oneSetOfStopWords = set(stopwords.words('english')+['``',"''",'...','nbsp','br','/div','div'])



def CleanText(givenText):

    reqText = givenText.lower()

    reqText = re.sub(r"=2e", "", reqText)

    reqText = re.sub(r"=2c", "", reqText)

    reqText = re.sub(r"\=", "", reqText)

    reqText = re.sub(r"news.website.http\:\/.*\/.*502503.stm.", "", reqText)

    reqText = re.sub(r"http://www.forcetacticalarmy.com","",reqText)

    reqText = re.sub(r"\'s", " ", reqText)

    reqText = re.sub(r"\'", " ", reqText)

    reqText = re.sub(r":", " ", reqText)

    reqText = re.sub(r"_", " ", reqText)

    reqText = re.sub(r"-", " ", reqText)

    reqText = re.sub(r"\'ve", " have ", reqText)

    reqText = re.sub(r"can't", "can not ", reqText)

    reqText = re.sub(r"n't", " not ", reqText)

    reqText = re.sub(r"i'm", "i am ", reqText)

    reqText = re.sub(r"\'re", " are ", reqText)

    reqText = re.sub(r"\'d", " would ", reqText)

    reqText = re.sub(r"\d", "", reqText)

    reqText = re.sub(r"\b[a-zA-Z]\b","", reqText)

    reqText = re.sub(r"[\,|\.|\&|\;|<|>]","", reqText)

    reqText = re.sub(r"\S*@\S*", " ", reqText)

    reqText = reqText.replace('_','')

    sentenceWords = []

    requiredWords = nltk.word_tokenize(reqText)

    for word in requiredWords:

        if word not in oneSetOfStopWords and word not in string.punctuation:

            sentenceWords.append(word)

    reqText = " ".join(sentenceWords)     

    return reqText



print (dataset.shape)

dataset = dataset[dataset['Text'].notnull()]

print (dataset.shape)
%%time

newDataset = dataset[dataset['Text'].notnull()][:5000]

newDataset['cleaned_text'] = newDataset.Text.apply(lambda x: CleanText(x))

newDataset.head()
newDataset['Class'].value_counts()
%%time

from nltk.corpus import stopwords

import string

oneSetOfStopWords = set(stopwords.words('english')+['``',"''",'...','nbsp','br','/div','div'])



totalWords = []

cleanedSentences = newDataset['cleaned_text'].values

for x in range(0,len(cleanedSentences)):

    tempWords = nltk.word_tokenize(cleanedSentences[x])

    for a in tempWords:

        totalWords.append(a)

wordfreqdist = nltk.FreqDist(totalWords)

mostcommon = wordfreqdist.most_common(100)

print(mostcommon)

import string

import re

from os import listdir

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer

from keras.utils.vis_utils import plot_model

from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.layers.embeddings import Embedding

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D
reqSentences = [list(x.split(" ")) for x in cleanedSentences]
from gensim.models import Word2Vec

from sklearn.decomposition import PCA



plt.figure(figsize=(12,12))

# train model



model = Word2Vec(reqSentences, min_count=1)

# save model

model.save('model.bin')

# fit a 2d PCA model to the vectors

X = model[model.wv.vocab]

#print (X[0])

pca = PCA(n_components=2)

result = pca.fit_transform(X)

# create a scatter plot of the projection

plt.scatter(result[:, 0], result[:, 1])

words = list(model.wv.vocab)

for i, word in enumerate(words):

    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()

cleanedSentences = newDataset['cleaned_text'].values

labels = newDataset['Class'].values

# integer encode the documents

vocab_size = 100

encoded_docs = [one_hot(d, vocab_size) for d in cleanedSentences]

# pad documents to a max length of 4 words

max_length = 100

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define the model



model = Sequential()

model.add(Embedding(vocab_size, 32, input_length=max_length))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

model.summary()

# fit the model

history = model.fit(padded_docs, labels, epochs=50, verbose=0)

# evaluate the model

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

print('Accuracy: %f' % (accuracy*100))

# fit a tokenizer

def create_tokenizer(lines):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer



# integer encode and pad documents

def encode_docs(tokenizer, max_length, docs):

    # integer encode

    encoded = tokenizer.texts_to_sequences(docs)

    # pad sequences

    padded = pad_sequences(encoded, maxlen=max_length, padding='post')

    return padded

# define the model

def define_model(vocab_size, max_length):

    model = Sequential()

    model.add(Embedding(vocab_size, 100, input_length=max_length))

    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # compile network

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize defined model

    model.summary()

    return model
np.random.seed(7)

trainDocuments = newDataset['cleaned_text'].values[:500]

ytrain = newDataset['Class'].values[:500]

Xtrain, Xtest, ytrain,ytest = train_test_split(trainDocuments, ytrain, test_size=0.2, random_state=1)

tokenizer = create_tokenizer(Xtrain)

vocabSize = len(tokenizer.word_index) + 1



print('Vocabulary size: %d' % vocabSize)

# calculate the maximum sequence length

max_length = max([len(s.split()) for s in trainDocuments])

print('Maximum length: %d' % max_length)



Xtrain = encode_docs(tokenizer, max_length, Xtrain)

Xtest = encode_docs(tokenizer, max_length, Xtest)

# define model

model = define_model(vocabSize, max_length)

# fit network

model.fit(Xtrain, ytrain, epochs=10, verbose=2)

_, acc = model.evaluate(Xtrain, ytrain, verbose=0)

print('Train Accuracy: %f' % (acc*100))

# evaluate model on test dataset

_, acc = model.evaluate(Xtest, ytest, verbose=0)

print('Test Accuracy: %f' % (acc*100))
def predict_sentiment(line,  tokenizer, max_length, model):

    # clean review

    padded = encode_docs(tokenizer, max_length, [line])

    # predict sentiment

    yhat = model.predict(padded, verbose=0)

    # retrieve predicted percentage and label

    print ("The prediction - ", yhat)

    percent_pos = yhat[0,0]

    if round(percent_pos) == 0:

        return (1-percent_pos), 'NEGATIVE'

    return percent_pos, 'POSITIVE'
text1 = ['Everyone', 'enjoy', 'film', 'I', 'love', 'recommended']

percent, sentiment = predict_sentiment(text1, tokenizer, max_length, model)

print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text1, sentiment, percent*100))

# test negative text

text2 = ['This', 'bad', 'movie', 'Do', 'watch', 'It', 'sucks']

percent, sentiment = predict_sentiment(text2, tokenizer, max_length, model)

print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text2, sentiment, percent*100))
testingWords = []

words = nltk.word_tokenize(text1)

for word in words:

    if word not in oneSetOfStopWords and word not in string.punctuation:

        testingWords.append(word)

print (testingWords)
dataset[dataset['Text'].notnull()][:10]
sentences = newDataset['cleaned_text'].values

labels = newDataset['Class'].values
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(sentences, labels, test_size = 0.25)
max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(Xtrain))

list_tokenized_train = tokenizer.texts_to_sequences(Xtrain)

list_tokenized_test = tokenizer.texts_to_sequences(Xtest)
maxlen = 200

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
import gensim.models.keyedvectors as word2vec

word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)

embed_size = 300
reqSentences = [row.split(" ") for row in sentences]
import gensim

model = gensim.models.Word2Vec(

    reqSentences,

    size=150,

    window=5,

    min_count=1,

    workers=10,

    iter=10)
model['business']
model.most_similar('dear')
model.most_similar('money')
embed_size = 150

embeddings_index = dict()

for word in model.wv.vocab:

    embeddings_index[word] = model.wv.word_vec(word)

print('Loaded %s word vectors.' % len(embeddings_index))

gc.collect()

#We get the mean and standard deviation of the embedding weights so that we could maintain the 

#same statistics for the rest of our own random generated weights. 

all_embs = np.stack(list(embeddings_index.values()))

emb_mean,emb_std = all_embs.mean(), all_embs.std()



nb_words = len(tokenizer.word_index)

#We are going to set the embedding size to the pretrained dimension as we are replicating it.

#the size will be Number of Words in Vocab X Embedding Size

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

gc.collect()



#With the newly created embedding matrix, we'll fill it up with the words that we have in both 

#our own dictionary and loaded pretrained embedding. 

embeddedCount = 0

for word, i in tokenizer.word_index.items():

    i-=1

    #then we see if this word is in glove's dictionary, if yes, get the corresponding weights

    embedding_vector = embeddings_index.get(word)

    #and store inside the embedding matrix that we will train later on.

    if embedding_vector is not None: 

        embedding_matrix[i] = embedding_vector

        embeddedCount+=1

print('total embedded:',embeddedCount,'common words')



del(embeddings_index)

gc.collect()



#finally, return the embedding matrix
import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

import matplotlib.pyplot as plt

%matplotlib inline

import gensim.models.keyedvectors as word2vec

import gc
inp = Input(shape=(maxlen, ))

x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)

x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

print (model.summary())
%%time

batch_size = 32

epochs = 4

model.fit(X_t,ytrain, batch_size=batch_size, epochs=epochs, validation_data=(X_te, ytest), verbose=2)