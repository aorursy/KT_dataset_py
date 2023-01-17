from string import punctuation

import os

import pickle



def normalize(txt):

    # removes punctuation and makes it small caps

    return ' '.join(word.strip(punctuation) for word in txt.split() if word.strip(punctuation)).lower()



root = "./text/"

all_files = []

for (dirname, subs, files) in os.walk(root):

    for fname in files:

        all_files.append(os.path.join(dirname, fname))



word_to_id = {}

words_count = 0

unique_words = set()



print(len(word_to_id))



count = 0

dataset = open('dataset.txt', 'w')



# We will use only the first 10 files (out of more than 7K)

for file_name in all_files[:10]:

    sentences = []

    file = open(file_name, 'rb')

    if count % 100 == 0:

        print('processing ', str(count), '/', str(len(all_files)))

    count += 1

    for line in file.readlines():

        text = line.decode('utf8')

        if text.startswith('<doc') or text.startswith('</doc') or text=='\n':

            continue

        tokens = normalize(text).split(' ')

        sentences.append(tokens)

        for t in tokens:

            if t not in word_to_id.keys():

                word_to_id[t] = words_count

                words_count += 1 



    # how far should we pay attention to the surrounding words

    window_size = 2 

    

    for sentence in sentences:

        for index, target in enumerate(sentence):

            start_neighbor = max(index - window_size, 0)

            end_neighbor = min(index + window_size, len(sentence)) + 1

            

            for neighbor in sentence[start_neighbor:end_neighbor]:

                        if neighbor != target:

                            target_id = word_to_id[target]

                            neighbor_id = word_to_id[neighbor]

                            dataset.write(str(target_id)+','+str(neighbor_id)+'\n')



dataset.close()

print('finished processing')



pickle.dump(word_to_id, open("word_to_id.pickle", "wb"))

del word_to_id



# randomising the data set before training

import random

with open('dataset.txt', 'r') as source:

    data = [ (random.random(), line) for line in source ]

data.sort()

with open('dataset_random.txt','w') as target:

    for _, line in data:

        target.write( line )

del data
import random

import tensorflow as tf

import pickle

import numpy as np

import math





word_to_id = pickle.load(open("../input/word2vec/word_to_id.pickle", "rb"))

vocab_size = len(word_to_id)



learning_rate = 5e-20

embedding_size = 50



# including batches was pointless in this case as the vocab_size is way too big (1M+)

x = tf.placeholder(tf.int32, shape=(1))

y = tf.placeholder(tf.int32, shape=(1))



one_hot = tf.one_hot(x, vocab_size)



# W1 actually holds the embedding vector aka word2vec, which we would use later on

W1 = tf.Variable(tf.random_normal([vocab_size, embedding_size], mean=0.0, stddev=1.0, dtype=tf.float32))

b1 = tf.Variable(tf.random_normal([embedding_size], mean=0.0, stddev=1.0, dtype=tf.float32))



hidden_layer = tf.nn.relu(tf.add(tf.matmul(one_hot, W1), b1))

dropout = tf.nn.dropout(hidden_layer, rate=0.1)



W2 = tf.Variable(tf.random_normal([embedding_size, vocab_size], mean=0.0, stddev=1.0, dtype=tf.float32))

b2 = tf.Variable(tf.random_normal([vocab_size], mean=0.0, stddev=1.0, dtype=tf.float32))



loss = tf.nn.softmax_cross_entropy_with_logits(

    logits = tf.add(tf.matmul(dropout, W2), b2),

    labels = (tf.one_hot(y, vocab_size)))



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gradients = optimizer.compute_gradients(loss)



trainable_vars = tf.trainable_variables()

grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars), 1)



train_op = optimizer.apply_gradients(zip(grads, trainable_vars))



sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)



counter = 0

iteration = 0

with open('../input/word2vec/dataset.txt', 'rb') as f:

    while True:

        entry = f.readline()

        counter += 1

        # make it at least somewhat random        

        # taking every 300-th example from around 6 Million examples  will result in around 20,000 iterations        

        if counter % 300 != 0:

            continue

        # There are about 130K entries, we will use roughly 10%.

        if entry: 

            values = entry.decode('utf8').split(',')

            inputs = np.asarray(int(values[0])).reshape((1))

            outputs = np.asarray(int(values[1])).reshape((1))

            

            sess.run(train_op, feed_dict={x: inputs, y: outputs})

        

            if iteration % 500 == 0:

                current_loss = sess.run(loss, feed_dict={x: inputs, y: outputs})

                print('iteration ' 

                      + str(iteration) + ' loss of current step is : ', current_loss)

                if math.isnan(current_loss):

                    break

            iteration += 1

        else:

            break



print('END')
word2vec = sess.run(W1)

pickle.dump(word2vec, open("word2vec.pickle", "wb"))
import tensorflow as tf

import numpy as np

from string import punctuation

import pickle





def normalize(txt):

    # removes punctuation and makes it small caps

    return ' '.join(word.strip(punctuation) for word in txt.split() if word.strip(punctuation)).lower()





word2vec = pickle.load(open("../input/word2vec/word2vec.pickle", "rb"))

word_to_id = pickle.load(open("../input/word2vec/word_to_id.pickle", "rb"))

        

max_len = 100

vocab_size = len(word_to_id)

embedding_size = word2vec[0].shape[0]





def words2vec(words):

    result = []

    for word in words:

        if word in word_to_id.keys():

            # simply skipping a word if it's not in the word2vec

            result.append(word2vec[word_to_id[word]])

    while len(result) < max_len:

        result.append(np.zeros((embedding_size)))

    return np.vstack(result[:max_len])

import tensorflow as tf



learning_rate = 0.00001

hidden_layer_1_size = embedding_size

hidden_layer_2_size = 16



x = tf.placeholder(tf.float32, (max_len, embedding_size))

y = tf.placeholder(tf.int32)



lstm = tf.contrib.rnn.BasicLSTMCell(embedding_size)

initial_state = state = lstm.zero_state(1, dtype=tf.float32)



for i in range(max_len):

    _, state = lstm(tf.reshape(x[i,:], (1, embedding_size)), state)

    

hidden_layer_1 = state



W1 = tf.Variable(tf.random_normal([hidden_layer_1_size ,hidden_layer_2_size]))

b1 = tf.Variable(tf.random_normal([hidden_layer_2_size]))



hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, W1), b1)



W2 = tf.Variable(tf.random_normal([hidden_layer_2_size, 2]))

b2 = tf.Variable(tf.random_normal([2]))



# in the last layer we have 2 classes - positive and negative review

# we calculate the loss by subtracting the labels from the positive class likelihood  

logits = tf.add(tf.matmul(hidden_layer_2, W2), b2)

predictions = tf.nn.softmax(logits)



loss = tf.nn.softmax_cross_entropy_with_logits(

    logits = logits,

    labels = (tf.one_hot(y, 2)))



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gradients = optimizer.compute_gradients(loss)



trainable_vars = tf.trainable_variables()

grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars), 1)



train_op = optimizer.apply_gradients(zip(grads, trainable_vars))



sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

saver = tf.train.Saver()



import bz2



train_set = "../input/amazonreviews/train.ft.txt.bz2"



# calculating_accuracy = False

iteration = 0

# total_val_set = 0

# accurate_val_set = 0



for line in bz2.BZ2File(train_set, "r"):

    tokens = normalize(line.decode('utf8')).split(' ')

    output = 1 if tokens[0] == 'label__2' else 0

    input = words2vec(tokens[1:])

#     if not calculating_accuracy:

    sess.run(train_op, feed_dict={x: input, y: output})

#     else:

#         total_val_set += 1

#         predictions = sess.run(predictions, feed_dict={x: input, y: output})

#         if predictions[1] == 1 and output == 1:

#             accurate_val_set += 1

#         if predictions[0] == 0 and output == 0:

#             accurate_val_set += 1

    

    if iteration % 10000 == 0:

        print('iteration '

              + str(iteration) + ' loss is : ',

              sess.run(loss, feed_dict={x: input, y: output}))

        saver.save(sess, 'sentiment-analysis')

        break

    iteration += 1

    if iteration > 20000:

        break

    # i.e. we have passed 90% of the data     

#         print('finished training')

#         calculating_accuracy = True



# print('accuracy over validation set:', accurate_val_set/total_val_set)
import bz2



test_set = "../input/amazonreviews/test.ft.txt.bz2"



iteration = 0

total_test_set = 0

accurate_test_set = 0

for line in bz2.BZ2File(test_set, "r"):

    tokens = normalize(line.decode('utf8')).split(' ')

    output = 1 if tokens[0] == 'label__2' else 0

    input = words2vec(tokens[1:])

    

    total_test_set += 1

    sess.run(predictions, feed_dict={x: input, y: output})

    if preds[1] == 1 and output == 1:

        accurate_test_set += 1

    if preds[0] == 0 and output == 0:

        accurate_test_set += 1

    

    if iteration % 100 == 0:

        print('accuracy over test set:', accurate_test_set/total_test_set)

        print('iteration ' + str(iteration))

    iteration += 1