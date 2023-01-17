# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/train.tsv',sep = '\t')

test = pd.read_csv('/kaggle/input/test.tsv',sep = '\t')
train.head(10)
train['Phrase'].str.len().mean()
train['Sentiment'].value_counts()


sns.barplot( x = ['2','3','1','4','0'],y=train['Sentiment'].value_counts())

plt.show()
train['Phrase'].str.len().max()
"""

    Convert data to proper format.

    1) Shuffle

    2) Lowercase

    3) Sentiments to Categorical

    4) Tokenize and Fit

    5) Convert to sequence (format accepted by the network)

    6) Pad

    7) Voila!

    """



#This method will be using the tokenized values



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



def format_data(train,test, max_features, maxlen):

    #This is to shuffle the data 

    train = train.sample(frac=1).reset_index(drop=True)

    

    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())

    test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())

    

    X = train['Phrase']

    print(X[1])

    test_X = test['Phrase']

    Y = to_categorical(train['Sentiment'].values)

    

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(X))

    

    X = tokenizer.texts_to_sequences(X)

    print(X[1])

    X = pad_sequences(X, maxlen=maxlen)

    test_X = tokenizer.texts_to_sequences(test_X)

    test_X = pad_sequences(test_X, maxlen=maxlen)



    return X,Y,test_X


#This method is just for tokenizing .

X,Y,test_X = format_data(train,test,max_features=10000,maxlen=125)
import re

import spacy

def cleanup_text_word2vec(docs, logging=False):

    sentences = []

    counter = 1

    nlp = spacy.load("en_core_web_sm")

    for doc in docs:

        if counter % 1000 == 0 and logging:

            print("Processed %d out of %d documents" % (counter, len(docs)))

        # Disable tagger so that lemma_ of personal pronouns (I, me, etc) don't getted marked as "-PRON-"

        doc = nlp(doc, disable=['tagger'])

        # Grab lemmatized form of words and make lowercase

        doc = " ".join([tok.lemma_.lower() for tok in doc])

        # Split into sentences based on punctuation

        doc = re.split("[\.?!;] ", doc)

        # Remove commas, periods, and other punctuation (mostly commas)

        doc = [re.sub("[\.,;:!?]", "", sent) for sent in doc]

        # Split into words

        doc = [sent.split() for sent in doc]

        sentences += doc

        counter += 1

    return sentences


all_text = np.concatenate((train['Phrase'], test['Phrase']), axis=0)

all_text = pd.DataFrame(all_text, columns=['Phrase'])

print('Number of total text documents:', len(all_text))
length = []

for i in range(len(train)):

    length.append((len(train['Phrase'][i].split())))



print(pd.Series(length).value_counts())

train_cleaned_word2vec = cleanup_text_word2vec(all_text['Phrase'], logging=True)
from gensim.models.word2vec import Word2Vec

def Word2Vec_(train):

    #train = train.sample(frac=1).reset_index(drop=True)

    

    #train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())

    #test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())

    

    

    #X = train['Phrase']

    #print(type(X))

    #test_X = test['Phrase']

    #Y = to_categorical(train['Sentiment'].values)

    

    #words = pd.concat([X,test_X],axis = 0)

    #print(len(words))

    wordvec_model = Word2Vec(train,size = 300, window=5, min_count=3, workers=4,sg = 1)

    print("Word to vec model created")

    print("%d unique words represented by %d dimensional vectors" % (len(wordvec_model.wv.vocab), 300))

    



    return wordvec_model 
wordvec_model = Word2Vec_(train_cleaned_word2vec)

print(wordvec_model.wv.most_similar(positive=['boy', 'girl'], negative=['man']))
def create_average_vec(doc,wordvec_model):

    average = np.zeros((300), dtype='float32')

    num_words = 0

    for word in doc.split():

        if word in wordvec_model.wv.vocab:

            average = np.add(average, wordvec_model[word])

            num_words += 1

    if num_words != 0:

        average = np.divide(average, num_words)

    return average
print(all_text['Phrase'][3],"\t",all_text['Phrase'][0])

print(train['Phrase'][3],"\t",train['Phrase'][0])
train_cleaned_vec = np.zeros((train.shape[0],300),dtype="float32")

print(len(train))

for i in range(len(train)):

    if i % 1000 == 0 :

            print("Processed %d out of %d documents" % (i, len(train)))

    train_cleaned_vec[i] = create_average_vec(all_text['Phrase'][i],wordvec_model)



print("Train word vector shape :" , train_cleaned_vec.shape)
train_cleaned_vec
from sklearn.model_selection import train_test_split

Y = to_categorical(train['Sentiment'].values)

X_train,X_test,Y_train,Y_test = train_test_split(train_cleaned_vec,Y,test_size = 0.25,random_state = 42)

print(X_train.shape," and ",Y_train.shape)

print(X_test.shape," and ",Y_test.shape)
from keras.layers import Dense, Embedding,LSTM

from keras.models import Sequential 

model = Sequential()



# Input / Embdedding

model.add(Embedding(300,100,mask_zero=True))

model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))

model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))



# Output layer

model.add(Dense(5, activation='sigmoid'))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=32, verbose=1)