import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np

import pandas as pd

import string

import nltk

from nltk.corpus import stopwords
data = pd.read_csv("/kaggle/input/my-data/googleplaystore_user_reviews.csv")
data.head()
data_reviews = pd.DataFrame()

data_reviews["Translated_Review"] =  data["Translated_Review"]
print("# Registers:"+str(data_reviews.shape[0]))

for column in data_reviews.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data_reviews[column]).values.ravel().sum()))
# Due to memory limitations we will only take the first 500 comments

corpus=data_reviews.dropna()
corpus.head()


def remove_stop_words(corpus):

    stops = stopwords.words('english')

    results = []

    for text in corpus:

        tmp = text.split(' ')

        for stop_word in stops:

            if stop_word in tmp:

                tmp.remove(stop_word)

        results.append(" ".join(tmp))

    

    return results
# Get array with all the texts without unnecesary words like is , and ,etc

corpus = remove_stop_words(corpus['Translated_Review'])
def normalize_text(texts, stops):

    texts = [x.lower() for x in texts]

    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    texts = [' '.join(word for word in x.split() if word not in (stops)) for x in texts]

    texts = [' '.join(x.split()) for x in texts]

    return texts



stops = stopwords.words('english')

corpus = normalize_text(corpus,stops)
def get_unique_words_in_corpus(corpus):

    words = []

    for text in corpus:

        for word in text.split(' '):

            words.append(word)

    return set(words) # Remove duplicates
words = get_unique_words_in_corpus(corpus)
#words.remove('') #Remove empty
words_dictionary = enumerate(words) # Each word has one ID

word2int = {}

for i,word in words_dictionary:

    word2int[word] = i 
# Separate each word in a sentence

sentences = []

for sentence  in corpus:

    sentences.append(sentence.split())
# Get the neighbors for each word

WINDOW_SIZE = 2



# Get all the neighbors for each word

words_with_neighbors = []

for sentence in sentences:

    for idx,word in enumerate(sentence):

        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] : 

            if neighbor != word:

                words_with_neighbors.append([word, neighbor])
skip_grams = pd.DataFrame(words_with_neighbors,columns=['input','target'])
skip_grams.head()
# The lenght for the one hot encoding is the total count of words in the corpus

ONE_HOT_DIM = len(words) 

def to_one_hot_encoding(data_point_index):

    one_hot_encoding = np.zeros(ONE_HOT_DIM)

    one_hot_encoding[data_point_index] = 1

    return one_hot_encoding
# making placeholders for X_train and Y_train

x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

y_target = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))



EMBEDDING_DIM = 100 # Number of neurons in hidden layer



# hidden layer: which represents word vector eventually

W1 = tf.Variable(tf.random_uniform([ONE_HOT_DIM, EMBEDDING_DIM],-1,1))

b1 = tf.Variable(tf.random_uniform([1],-1,1)) #bias

hidden_layer = tf.add(tf.matmul(x,W1), b1)



# output layer

W2 = tf.Variable(tf.random_uniform([EMBEDDING_DIM, ONE_HOT_DIM],-1,1))

b2 = tf.Variable(tf.random_uniform([1],-1,1))

prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))



# loss function: cross entropy

loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(prediction), axis=[1]))



# training operation

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init) 



# Convert to arrays

X_train_array = skip_grams['input'].to_numpy()

Y_train_array = skip_grams['target'].to_numpy()



iteration = 2000

batch_size = 150

for i in range(iteration):

    # input is X_train which is one hot encoded word

    # label is Y_train which is one hot encoded neighbor word

    ## Seleccionamos el batch de manera random

    rand_idx = np.random.choice(len(skip_grams), size=batch_size)



    rand_x = X_train_array[rand_idx]

    rand_y = Y_train_array[rand_idx]

    

    rand_x_in_one_hot_encoding =np.asarray([to_one_hot_encoding(word2int[ x ]) for x in rand_x])

    rand_y_in_one_hot_encoding = np.asarray([to_one_hot_encoding(word2int[ y ]) for y in rand_y])

    

    sess.run(train_op, feed_dict={x: rand_x_in_one_hot_encoding, y_target: rand_y_in_one_hot_encoding})

    if i % 100 == 0:

        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: rand_x_in_one_hot_encoding, y_target: rand_y_in_one_hot_encoding}))

      



# Now the hidden layer (W1 + b1) is actually the word look up table

vectors = sess.run(W1 + b1)

print(vectors)