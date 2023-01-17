# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/bbc-fulltext-and-category/bbc-text.csv")

data.head()

train_i = data['text']

label = data['category']
train_i.head()
import seaborn as sns

sns.set(rc={'figure.figsize':(10,10)})

sns.countplot(label)
from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(train_i,label,test_size = 0.3,random_state = 42)
print(X_train[0])
# remove punctuation marks

punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'



X_train = X_train.apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

X_test = X_test.apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))



# convert text to lowercase

X_train = X_train.str.lower()

X_test  = X_test.str.lower()



# remove numbers

X_train = X_train.str.replace("[0-9]", " ")

X_test  = X_test.str.replace("[0-9]", " ")



# remove whitespaces

X_train = X_train.apply(lambda x:' '.join(x.split()))

X_test  = X_test.apply(lambda x: ' '.join(x.split()))
#apply word Normalisation :

import spacy

nlp = spacy.load('en', disable = ['parser','ner'])

def lemmetization(text):

    output = []

    for i in text:

        s = [token.lemma_ for token in nlp(i)]

        output.append(''.join(s))

    return output
#X_train = lemmetization(X_train)

#X_test = lemmetization(X_test)
import keras



tokenizer = keras.preprocessing.text.Tokenizer(num_words = None,char_level = False)



tokenizer.fit_on_texts(X_train)

vocab = len(tokenizer.word_counts)+1

train_sequence = tokenizer.texts_to_sequences(X_train)

test_sequence = tokenizer.texts_to_sequences(X_test)

max_word = 1000

train = keras.preprocessing.sequence.pad_sequences(train_sequence,maxlen=max_word)

test = keras.preprocessing.sequence.pad_sequences(test_sequence,maxlen=max_word)


print(vocab)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(y_train)

train_label  = le.transform(y_train)

test_label = le.transform(y_test)

# Converts the labels to a one-hot representation

num_classes = np.max(train_label) + 1

# Converts the labels to a one-hot representation

num_classes = np.max(train_label) + 1

train_label = keras.utils.to_categorical(train_label, num_classes)

test_label = keras.utils.to_categorical(test_label, num_classes)
print(train_label.shape)

print(test_label.shape)
d = dict(zip(le.classes_, le.transform(le.classes_)))

print (d)

category = []

for key in d:

    category.append(key)

print(category)

    
print(train.shape)

print(test.shape)
from sklearn.feature_extraction.text import TfidfTransformer

idf = TfidfTransformer(smooth_idf = False)

idf.fit(train)

train_idf = idf.transform(train)

test_idf = idf.transform(test)
print(train_idf.shape)
#from sklearn.multiclass import OneVsRestClassifier

#from sklearn.svm import SVC

#classif = OneVsRestClassifier(SVC(kernel='linear'))

#classif.fit(train_idf, train_label)
#score = classif.score(test_idf,test_label, sample_weight=None)

#print(score)
#pred = classif.predict(test_idf)
#from sklearn.naive_bayes import MultinomialNB

#mnb = MultinomialNB()

#mnb.fit(train_idf,train_label)
#score = mnb.score(test_idf,test_label, sample_weight=None)

#print(score)
output = train_label.shape

print(output)

import tensorflow as tf

model = keras.Sequential([

   

    keras.layers.Embedding(vocab,40,input_length = 1000),

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation=tf.nn.relu),

    keras.layers.Dense(5, activation=tf.nn.softmax)

])
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
batch_size = 32

epochs = 20

drop_ratio = 0.5

history = model.fit(train, train_label,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_split=0.1)

score =  model.evaluate(test,test_label,batch_size = batch_size,verbose = 1)

print(score[0])

print(score[1])
predictor = model.predict(test,batch_size = batch_size,verbose = 1)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer

# Count Vectors as features

# create a count vectorizer object 

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(train_i)



# transform the training and validation data using count vectorizer object

xtrain_count =  count_vect.transform(X_train)

xvalid_count =  count_vect.transform(X_test)