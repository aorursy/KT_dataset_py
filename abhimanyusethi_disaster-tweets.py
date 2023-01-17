# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

import re

from keras.models import Model

from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.initializers import glorot_uniform



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.DataFrame()

test = pd.DataFrame()
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.sample(10)
test.sample(5)
#Rectifying the mislabelled data

indices = [4415, 4400, 4399,4403,4397,4396, 4394,4414, 4393,4392,4404,4407,4420,4412,4408,4391,4405]

train.loc[indices]
train.loc[indices,'target'] = 0
indices = [6840,6834,6837,6841,6816,6828,6831]

train.loc[indices]
train.loc[indices,'target'] = 0
indices = [601,576,584,608,606,603,592,604,591, 587]

train.loc[indices]
train.loc[indices,'target'] = 1
indices = [3913,3914,3936,3921,3941,3937,3938,3136,3133,3930,3933,3924,3917]

train.loc[indices]
train.loc[indices,'target'] = 0
indices = [246,270,266,259,253,251,250,271]

train.loc[indices]
train.loc[indices,'target'] = 0
indices = [6119,6122,6123,6131,6160,6166,6167,6172,6212,6221,6230,6091,6108]

train.loc[indices]
train.loc[indices,'target'] = 0
indices = [7435,7460,7464,7466,7469,7475,7489,7495,7500,7525,7552,7572,7591,7599]

train.loc[indices]
train.loc[indices,'target'] = 0
#Cleaning text this point onwards

def remove_url(sentence):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',sentence)
def remove_at(sentence):

    at = re.compile(r'@\S+')

    return at.sub(r'',sentence)
def remove_html(sentence):

    html = re.compile(r'<.*?>')

    return html.sub(r'', sentence)
def remove_emoji(sentence):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    

    return emoji_pattern.sub(r'', sentence)
def clean_text(data):

    data['text'] = data['text'].apply(lambda x : remove_url(x))

    data['text'] = data['text'].apply(lambda x : remove_at(x))

    data['text'] = data['text'].apply(lambda x : remove_html(x))

    data['text'] = data['text'].apply(lambda x : remove_emoji(x))

    

    return data

train = clean_text(train)

test = clean_text(test)
train.loc[range(30,40)]
#tokenizing sentences to generate our dictionary

def define_tokenizer(train_sentences,test_sentences):

    sentences = pd.concat([train_sentences, test_sentences])

    

    tokenizer = tf.keras.preprocessing.text.Tokenizer()

    tokenizer.fit_on_texts(sentences)

    

    return tokenizer
def encode(sentences, tokenizer):

    encoded_sentences = tokenizer.texts_to_sequences(sentences)

    encoded_sentences = tf.keras.preprocessing.sequence.pad_sequences(encoded_sentences, 33,padding='post')

    

    return encoded_sentences
tokenizer = define_tokenizer(train['text'], test['text'])



encoded_sentences = encode(train['text'], tokenizer)

encoded_test_sentences = encode(test['text'], tokenizer)
print(encoded_sentences[0])
tokenizer.word_index["our"]

#print(tokenizer.word_index)

#print(tokenizer.word_index.items())
#Importing downloaded GloVe word embeddings

embedding_dict = {}



with open('/kaggle/input/glove-6b-50d/glove.6B.50d.txt','r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        vectors = np.asarray(values[1:],'float32')

        embedding_dict[word] = vectors

        

f.close()
embedding_dict["hello"]
#making embedding matrix and lining up values so that embedding matrix and our tokenizer dictionary match up

num_words = len(tokenizer.word_index)+1

emb_matrix = np.zeros((num_words,50))



for word, index in tokenizer.word_index.items():

    if index > num_words:

        continue

    

    emb_vec = embedding_dict.get(word)

    

    if emb_vec is not None:

        emb_matrix[index] = emb_vec
print(emb_matrix[1])
#def to_one_hot(encoded_sentences):

 #   for i in range(encoded_sentences.shape[0]):

  #      sentence = encoded_sentences[i]

   #     encoded_sentences[i] = tf.keras.utils.to_categorical(sentence,num_classes=len(tokenizer.word_index)+1)

    #return encoded_sentences
#encoded_sentences_oh = to_one_hot(encoded_sentences)
print(train['target'])
Y = np.transpose(np.asarray(train['target'],'int'))

print(Y)
def sentence_to_avg(sentence,max_len,emb_matrix):

    avg = np.zeros((1,50))

    for i in range(max_len):

        avg += (emb_matrix[sentence[i]])/33

    

    return avg

        
first_sentence = encoded_sentences[0]

print(encoded_test_sentences.shape)
print(first_sentence)
#example

#average = sentence_to_avg(first_sentence,emb_matrix)

#print(average.shape)
def model(X, Y,emb_matrix, learning_rate = 0.01, num_iterations = 400):

    """

    Model to train word vector representations in numpy.

    

    Arguments:

    X -- input data, numpy array of sentences as strings, of shape (m, 1)

    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)

    learning_rate -- learning_rate for the stochastic gradient descent algorithm

    num_iterations -- number of iterations

    

    Returns:

    pred -- vector of predictions, numpy-array of shape (m, 1)

    W -- weight matrix of the softmax layer, of shape (n_y, n_h)

    b -- bias of the softmax layer, of shape (n_y,)

    """



    # Define number of training examples

    m = Y.shape[0]                          # number of training examples

    n_y = 1                                # number of classes  

    n_h = 50                                # dimensions of the GloVe vectors 

    

    # Initialize parameters using Xavier initialization

    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)

    b = np.zeros((n_y,))

    

    

    # Optimization loop

    for t in range(num_iterations): # Loop over the number of iterations

        for i in range(m):          # Loop over the training examples

            

            ### START CODE HERE ### (≈ 4 lines of code)

            # Average the word vectors of the words from the i'th training example

            avg = np.transpose(sentence_to_avg(X[i],33, emb_matrix))



            # Forward propagate the avg through the softmax layer

            z = np.dot(W,avg) + b

            a = tf.math.sigmoid(z)



            # Compute cost using the ith training label's one hot representation and "A" (the output of the softmax)

            cost = -np.sum(Y[i]* np.log(a))

            ### END CODE HERE ###

            

            # Compute gradients 

            dz = a - Y[i]

            dW = np.dot(np.reshape(dz,(n_y,1)), np.reshape(avg,(1, n_h)))

            db = dz



            # Update parameters with Stochastic Gradient Descent

            W = W - learning_rate * dW

            b = b - learning_rate * db

        

        if t % 5 == 0:

            print("Epoch: " + str(t) + " --- cost = " + str(cost))

            #pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py



    return  W, b
W,b = model(encoded_sentences, Y,emb_matrix, 0.01, 50)
def predict(X,W,b,emb_matrix):

    m = X.shape[0]

    pred = []

    for i in range (m):

        avg = np.transpose(sentence_to_avg(X[i],31, emb_matrix))

        z = np.dot(W,avg) + b

        a = tf.math.sigmoid(z)

        if a>0.5:

            a = 1

        else:

            a=0

        pred.append(a)

    return pred
#pred_array =  np.asarray(predict(encoded_test_sentences,W,b,emb_matrix))
#print(pred_array.shape)
#sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

#print(sample)

#sample['target'] = pred_array
#sample.to_csv('submission.csv', index=False)
sentences = Input((33,),dtype='int32')

embedding_layer = Embedding(len(tokenizer.word_index)+1,50)

embedding_layer.build((None,))

embedding_layer.set_weights([emb_matrix])

embeddings = embedding_layer(sentences)



X = Bidirectional(LSTM(units=128,return_sequences=True))(embeddings)

X = Dropout(0.6)(X)

X = Bidirectional(LSTM(units=128,return_sequences=False))(X)

X = Dropout(0.6)(X)

X = Dense(units=1)(X)

X = Activation('sigmoid')(X)



model = Model(inputs = sentences,outputs = X)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(encoded_sentences,Y , epochs = 20, batch_size = 256, shuffle=True)
out = model.predict(encoded_test_sentences)
print(out)
out[out>0.5] = 1

out[out<=0.5] = 0
output = np.asarray(out,'int')
print(output)
sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sample['target'] = output

sample.to_csv('submission.csv', index=False)