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
import tensorflow as tf

import pandas as pd       

train0 = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, \

                    delimiter="\t", quoting=3)

test = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", \

                   quoting=3 )

train1=pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

train1['sentiment'] = train1['sentiment'].map({'positive': 1, 'negative': 0})

train0=train0.drop('id',axis=1)

train= pd.concat([train0, train1]).reset_index(drop=True)

train.shape
import nltk

#nltk.download('all')  # Download text data sets, including stop words

from nltk.corpus import stopwords # Import the stop word list
from bs4 import BeautifulSoup             

import re

def review_to_words( raw_review ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review).get_text() 

    #

    # 2. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    #

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))  





# Initialize the BeautifulSoup object on a single movie review     

cleaned_reviews=[]

for review in train["review"]:

    cleaned_reviews.append(review_to_words( review ))



all_text = ' '.join(cleaned_reviews)

words = all_text.split()





from collections import Counter

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



reviews_ints = []

for each in cleaned_reviews:

    reviews_ints.append([vocab_to_int[word] for word in each.split()])







labels = np.array(train['sentiment'])



non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]

labels = np.array([labels[ii] for ii in non_zero_idx])
seq_len = 1000

features = np.zeros((len(reviews_ints), seq_len), dtype=int)

for i, row in enumerate(reviews_ints):

    features[i, -len(row):] = np.array(row)[:seq_len]



split_frac = 0.95

split_idx = int(len(features)*0.8)

train_x, val_x = features[:split_idx], features[split_idx:]

train_y, val_y = labels[:split_idx], labels[split_idx:]



test_idx = int(len(val_x)*0.5)

val_x, test_x = val_x[:test_idx], val_x[test_idx:]

val_y, test_y = val_y[:test_idx], val_y[test_idx:]



print("\t\t\tFeature Shapes:")

print("Train set: \t\t{}".format(train_x.shape), 

      "\nValidation set: \t{}".format(val_x.shape),

      "\nTest set: \t\t{}".format(test_x.shape))

lstm_size = 256

lstm_layers = 1

batch_size_ = 500

learning_rate = 0.001



n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1



# Create the graph object

graph = tf.Graph()

# Add nodes to the graph

with graph.as_default():

    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')

    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    batch_size = tf.placeholder(tf.int32,[] ,name='batch_size')



# Size of the embedding vectors (number of units in the embedding layer)

embed_size = 500 



with graph.as_default():

    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))

    embed = tf.nn.embedding_lookup(embedding, inputs_)



with graph.as_default():

    # Your basic LSTM cell

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    

    # Add dropout to the cell

    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    

    # Stack up multiple LSTM layers, for deep learning

    cell = tf.contrib.rnn.MultiRNNCell([drop  for i in range(lstm_layers)])

    

    # Getting an initial state of all zeros

    initial_state = cell.zero_state(batch_size, dtype=tf.float32)



with graph.as_default():

    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,

                                             initial_state=initial_state)

with graph.as_default():

    #predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)

    ann= tf.contrib.layers.fully_connected(outputs[:, -1], 256, activation_fn=tf.sigmoid)

    predictions = tf.contrib.layers.fully_connected(ann, 1, activation_fn=tf.sigmoid)

    cost = tf.losses.mean_squared_error(labels_, predictions)

    

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



with graph.as_default():

    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



def get_batches(x, y, batch_size=100):

    

    n_batches = len(x)//batch_size

    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]

    for ii in range(0, len(x), batch_size):

        yield x[ii:ii+batch_size], y[ii:ii+batch_size]    

        

        

epochs = 10

with graph.as_default():

    #saver = tf.train.Saver()

    saver = tf.train.Saver(var_list=tf.trainable_variables())

with tf.Session(graph=graph) as sess:

    sess.run(tf.global_variables_initializer())

    iteration = 1

    for e in range(epochs):

        state = sess.run(initial_state,feed_dict={batch_size:batch_size_})

        

        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size_), 1):

            feed = {inputs_: x,

                    labels_: y[:, None],

                    keep_prob: 0.5,

                    initial_state: state,

                    batch_size:batch_size_ }

            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            

            if iteration%5==0:

                print("Epoch: {}/{}".format(e, epochs),

                      "Iteration: {}".format(iteration),

                      "Train loss: {:.3f}".format(loss))



            if iteration%25==0:

                val_acc = []

                val_state = sess.run(initial_state,feed_dict={batch_size:batch_size_})

                for x, y in get_batches(val_x, val_y, batch_size_):

                    feed = {inputs_: x,

                            labels_: y[:, None],

                            keep_prob: 1,

                            initial_state: val_state,

                            batch_size:batch_size_}

                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)

                    val_acc.append(batch_acc)

                print("Val acc: {:.3f}".format(np.mean(val_acc)))

            iteration +=1

    saver.save(sess, "checkpoints/sentiment.ckpt")

    #saver.save(sess, "sentiment_model")
cleaned_test_reviews=[]

for review in test["review"]:

    cleaned_test_reviews.append(review_to_words( review ))





reviews_test_ints = []

for each in cleaned_test_reviews:

    reviews_test_ints.append([vocab_to_int[word] for word in each.split()])

    

non_zero_idx = [ii for ii, review in enumerate(reviews_test_ints) if len(review) != 0]

reviews_test_ints = [reviews_test_ints[ii] for ii in non_zero_idx]    



seq_len = 1000

features_test = np.zeros((len(reviews_test_ints), seq_len), dtype=int)

for i, row in enumerate(reviews_test_ints):

    features_test[i, -len(row):] = np.array(row)[:seq_len]

X_test=features_test

X_test.shape

preds=[]

with tf.Session(graph=graph) as sess:

    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    #saver.restore(sess, tf.train.latest_checkpoint('./'))



    test_state = sess.run(initial_state,feed_dict={batch_size:batch_size_})

    for i in range(0,X_test.shape[0],batch_size_):

        x=X_test[i:i+batch_size_]

        

        feed = {inputs_: x,                

                keep_prob: 1,

                initial_state: test_state,

                batch_size:1  

               }

        pred = sess.run(predictions, feed_dict=feed)

        preds.extend(pred)

sentiments=[1 if p>0.5 else 0 for p in preds ]

test_sub=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/sampleSubmission.csv')

test_sub=test_sub.drop('sentiment',axis=1)

test_sub['sentiment']=np.array( sentiments)

test_sub.to_csv('sampleSubmission06.csv',index=None)

import os

print(os.getcwd())
from IPython.display import FileLink

FileLink(r'sampleSubmission06.csv')