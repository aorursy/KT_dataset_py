# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from nltk.corpus import stopwords

from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split

import datetime

from time import time

import itertools

from keras.preprocessing.sequence import pad_sequences

from keras.models import Model

from keras.layers import Input, Embedding,LSTM,Lambda

import keras.backend as K

from keras.optimizers import Adadelta

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## Loading the data set 

train=pd.read_csv('/kaggle/input/question-pairs-dataset/questions.csv')## quora question pair

test= pd.read_csv("/kaggle/input/text-similarity/Text_Similarity_Dataset.csv")

Embedding_vectors='/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'### pre-train word embedding
train.drop(['qid1','qid2'],axis=1,inplace=True)
train.rename(columns={'id':'Unique_Id','question1':'text1','question2':'text2','is_duplicate':'similarity'},inplace=True)
train.head()
test.head()
test.shape
train=train[:5000]
train.shape
## Data Preprocessing

stops = set(stopwords.words('english'))

def text_to_word_list(text):

    ''' Pre \process and convert texts to a list of words '''

    text = str(text)

    text = text.lower()

    # Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

# Prepare embedding

vocabulary = dict()

inverse_vocabulary = ['<unk>']  # '<unk>'  is only a placeholder for the [0, 0, ....0] embedding

#KeyedVectors and is essentially a mapping between entities and vectors

word2vec = KeyedVectors.load_word2vec_format(Embedding_vectors, binary=True)##loading Embedding

##preparing both train and test data for tranining by creating embedding of each words in the text1 and text2

questions_cols = ['text1', 'text2']

# Iterate over the questions only of both training and test datasets

for dataset in [train, test]:

    for index, row in dataset.iterrows():



        # Iterate through the text of both questions of the row

        for question in questions_cols:



            q2n = []  # q2n -> question numbers representation

            for word in text_to_word_list(row[question]):

                # Check for unwanted words

                if word in stops and word not in word2vec.vocab:

                    continue

                if word not in vocabulary:

                    vocabulary[word] = len(inverse_vocabulary)

                    q2n.append(len(inverse_vocabulary))

                    inverse_vocabulary.append(word)

                else:

                    q2n.append(vocabulary[word])

            # Replace questions as word to question as number representation

            dataset.set_value(index, question, q2n)
# creating embeding with 300 dimensions    

embedding_dim = 300

embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix

embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix

for word, index in vocabulary.items():

    if word in word2vec.vocab:

        embeddings[index] = word2vec.word_vec(word)



del word2vec#free space by deleting variable
train.head()
test.head()
## findinf max length of text1 and text2 in both train an text for padding to make the both sentences length equal.

max_seq_length = max(train.text1.map(lambda x: len(x)).max(),

                     train.text2.map(lambda x: len(x)).max(),

                     test.text1.map(lambda x: len(x)).max(),

                     test.text2.map(lambda x: len(x)).max())



# Split to train validation

validation_size = 1000

training_size = len(train) - validation_size



X = train[questions_cols]

Y = train['similarity']



X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)



# Split to dicts

X_train = {'left': X_train.text1, 'right': X_train.text2}

X_validation = {'left': X_validation.text1, 'right': X_validation.text2}

X_test = {'left': test.text1, 'right': test.text2}



# Convert labels to their numpy representations

Y_train = Y_train.values

Y_validation = Y_validation.values



# Zero padding

for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):

    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)



""" assert help to check if the program is running smoothly if returns nothing 

if it is true but raises and error if it is false and stops further execution"""

assert X_train['left'].shape == X_train['right'].shape

assert len(X_train['left']) == len(Y_train)


# Model variables

n_hidden = 50

gradient_clipping_norm = 1.25

batch_size = 70

n_epoch = 10



def exponent_neg_manhattan_distance(left, right):

    ''' Helper function for the similarity estimate of the LSTMs outputs'''

    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))



# The visible layer

left_input = Input(shape=(max_seq_length,), dtype='int32')

right_input = Input(shape=(max_seq_length,), dtype='int32')



embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)



# Embedded version of the inputs

encoded_left = embedding_layer(left_input)

encoded_right = embedding_layer(right_input)



# Since this is a siamese network, both sides share the same LSTM

shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)

right_output = shared_lstm(encoded_right)



# Calculates the distance as defined by the MaLSTM model

malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])



# Pack it all up into a model

malstm = Model([left_input, right_input], [malstm_distance])



# Adadelta optimizer, with gradient clipping by norm

optimizer = Adadelta(clipnorm=gradient_clipping_norm)



malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training

training_start_time = time()



malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,

                            validation_data=([X_validation['left'], X_validation['right']], Y_validation))



print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
# preparing the text data for prediction
# Split to dicts

X_test = {'left': test.text1, 'right': test.text2}

# Zero padding

for dataset, side in itertools.product([X_test], ['left', 'right']):

    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)



assert X_test['left'].shape == X_test['right'].shape

result=malstm.predict([X_test['left'], X_test['right']])
prediction=result ## copying the variable so that we can reuse if any thing goes wrong
prediction=prediction.tolist()### ndarry to list
#Here we flatten the list of list to a single list

flatten = itertools.chain.from_iterable

prediction=list(flatten(prediction))

pred = [round(num) for num in prediction]#Here we round the manhattan distance of the prediction which is between 0 and 1
final= pd.DataFrame(pred, columns =['similarity'])
submission=result = pd.concat([test['Unique_ID'], final['similarity']], axis=1, sort=False)
submission.to_csv('submision.csv')## prediction to csv 
### pickling the model for future use

from sklearn.externals import joblib  

joblib.dump(malstm, 'model.pkl')