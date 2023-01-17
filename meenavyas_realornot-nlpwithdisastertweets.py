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
#id - a unique identifier for each tweet

#text - the text of the tweet

#location - the location the tweet was sent from (may be blank)

#keyword - a particular keyword from the tweet (may be blank)

#PREDICT

#target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



print(train_df.shape) # 7613, 5 , type is <class 'pandas.core.frame.DataFrame'>

print(test_df.shape) # 3263, 4
print(train_df.head(3)) # id, keyword, location, text, target
print(test_df.head(3)) # id, keyword, location, text
print(train_df.isna().sum().sum())

print(test_df.isna().sum().sum())

train_df["text"].fillna("", inplace = True) 

test_df["text"].fillna("", inplace = True) 

print(train_df.head(3)) # id, keyword, location, text, target

print(test_df.head(3)) # id, keyword, location, text
# get unique characters in train text

unique_chars = {}

train_text_col =  train_df['text']

for idx in range(0,train_text_col.shape[0]):

    sentence = train_text_col[idx]

    for ch in sentence:

        lower_char = ch.lower()

        unique_chars[lower_char] = 1



print(sorted(unique_chars.keys()))

# contains '0' to '9'

# contains 'a' to 'z'

# contains whitespace ' ', '\n'
import re

def my_clean(df, column_name):

    # remove punctuation marks

    punctuation = "!\"'#$£%&()*+-/÷:;<=>?@[\\]_^`´{|}~©¼«¬" 

    #punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

    df[column_name] = df[column_name].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

    # convert text to lowercase

    df[column_name] = df[column_name].str.lower()

    # remove numbers

    df[column_name] = df[column_name].str.replace("[0-9]", " ")

    # remove whitespaces

    df[column_name] = df[column_name].str.replace("\n", " ")

    df[column_name] = df[column_name].apply(lambda x:' '.join(x.split()))

    # remove URL's from train and test

    df[column_name] = df[column_name].apply(lambda x: re.sub(r'http\S+', '', x))

    return df
train_df = my_clean(train_df,'text')

# Remaining [' ', ',', '.',  '\x89', '\x9d', '¡', '¢', '¤', '¨', 'ª', 'â', 'ã', 'å', 'ç', 'è', 'ê', 'ì', 'ï', 'ñ', 'ò', 'ó', 'û', 'ü']

test_df = my_clean(test_df,'text')
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])



# function to lemmatize text

def lemmatization(texts):

    output = []

    for i in texts:

        s = [token.lemma_ for token in nlp(i)]

        output.append(' '.join(s))

    return output
train_df['text'] = lemmatization(train_df['text'])

test_df['text'] = lemmatization(test_df['text']) 
import tensorflow.compat.v1 as tf

print(tf.__version__)

tf.disable_eager_execution()
import tensorflow_hub as hub

print(hub.__version__)
## refer https://www.kaggle.com/c/google-quest-challenge/discussion/122389

## Temporary failure in name resolution - internet On

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True) 

## getting error RuntimeError: Exporting/importing meta graphs is not supported when eager execution is enabled. 

## No graph exists when eager execution is enabled.

## So took compat.v1 version of tf
# get embedding shape

x = ["Roasted ants are a popular snack in Columbia"]

## x = ["the cat is on the mat", "dogs are in the fog"]

# Extract ELMo features 

embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

print(embeddings.shape) # should be 1024?
def elmo_vectors(x):

    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        sess.run(tf.tables_initializer())

        # return average of ELMo features

        return sess.run(tf.reduce_mean(embeddings,1))
list_train = [train_df[i:i+100] for i in range(0,train_df.shape[0],100)]

list_test = [test_df[i:i+100] for i in range(0,test_df.shape[0],100)]
# Extract ELMo embeddings - text

elmo_train_text = [elmo_vectors(x['text']) for x in list_train]

elmo_test_text = [elmo_vectors(x['text']) for x in list_test]
print(elmo_train_text.head(3)) # id, keyword, location, text, target

print(elmo_test_text.head(3)) # id, keyword, location, text, target
elmo_train_new = np.concatenate(elmo_train_text,  axis = 0)

elmo_test_new = np.concatenate(elmo_test_text, axis = 0)
print(type(elmo_train_new))

print(elmo_train_new[0]) # just ome numbers

print(elmo_test_new[0]) # just ome numbers
from sklearn.model_selection import train_test_split

xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new,  train_df['target'],  

                                                  random_state=42, test_size=0.2)
print(type(xtrain)) # ndarray

print(type(xtrain[0])) # ndarray

print(xtrain[0])



print(type(xvalid))

print(type(xvalid[0]))

print(xvalid[0])



print(type(ytrain)) # Series

print(type(yvalid)) # Series

print(ytrain)

print(yvalid)
#from sklearn.linear_model import LogisticRegression

#from sklearn.metrics import f1_score



#lreg = LogisticRegression()

#lreg.fit(xtrain, ytrain)

#preds_valid = lreg.predict(xvalid)

#f1_score(yvalid, preds_valid)

# make predictions on test set

#preds_test = lreg.predict(elmo_test_new)



# prepare submission dataframe

#sub = pd.DataFrame({'id':test_df['id'], 'target':preds_test})



# write predictions to a CSV file

#sub.to_csv("submission.csv", header=True, index=False)
from sklearn.neighbors import NearestNeighbors

tree = NearestNeighbors(algorithm='auto', metric='cosine')

tree.fit(xtrain)
print(tree)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score

k = KNeighborsClassifier(algorithm='auto', metric='cosine')

k.fit(xtrain, ytrain)
preds_valid = k.predict(xvalid)
f1_score(yvalid, preds_valid) # 0.7539
k1 = KNeighborsClassifier(n_neighbors=2, algorithm='auto', metric='cosine')

k1.fit(xtrain, ytrain)

preds_valid1 = k1.predict(xvalid)

f1_score(yvalid, preds_valid1) # worse 0.694
k2 = KNeighborsClassifier()

k2.fit(xtrain, ytrain)

preds_valid2 = k.predict(xvalid)

f1_score(yvalid, preds_valid2) # 0.7539
# make predictions on test set

preds_test = k.predict(elmo_test_new)
# prepare submission dataframe

sub = pd.DataFrame({'id':test_df['id'], 'target':preds_test})



# write predictions to a CSV file

sub.to_csv("submission.csv", header=True, index=False)