# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.datasets import cifar10



(x_train, y_train),(x_test, y_test) = cifar10.load_data()



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
import tensorflow as tf



tf.reset_default_graph() # clean the graph we built before



Num_Classes = 10



X = tf.placeholder(tf.float32, [None, 32, 32, 3]) # [N,H,W,C]

Y = tf.placeholder(tf.int64, [None,1])



X_extend = tf.reshape(X, [-1, 32,32,3])

Y_onehot = tf.one_hot(indices = Y, depth = Num_Classes)





"""first convolutionb layer"""

conv1_w = tf.get_variable("conv1_w", [3,3,3,10], initializer= tf.random_normal_initializer(stddev=1e-2))



conv1_b = tf.get_variable("conv1_b", [10], initializer= tf.random_normal_initializer(stddev=1e-2))



conv1 = tf.nn.conv2d(X_extend, conv1_w, strides = [1,1,1,1], padding= 'SAME')+conv1_b

relu1= tf.nn.relu(conv1)

pool1 = tf.nn.max_pool(value=relu1, ksize= [1,2,2,1], strides = [1,2,2,1], padding= 'SAME')





"""Second convolutionb layer"""

conv2_w = tf.get_variable("conv2_w", [3,3,10,10], initializer= tf.random_normal_initializer(stddev=1e-2))



conv2_b = tf.get_variable("conv2_b", [10], initializer= tf.random_normal_initializer(stddev=1e-2))



conv2 = tf.nn.conv2d(pool1, conv2_w, strides = [1,1,1,1], padding= 'SAME')+conv2_b

relu2= tf.nn.relu(conv2)

pool2 = tf.nn.max_pool(value=relu2, ksize= [1,2,2,1], strides = [1,2,2,1], padding= 'SAME')





"""third convolutionb layer"""

conv3_w = tf.get_variable("conv3_w", [3,3,10,10], initializer= tf.random_normal_initializer(stddev=1e-2))



conv3_b = tf.get_variable("conv3_b", [10], initializer= tf.random_normal_initializer(stddev=1e-2))



conv3 = tf.nn.conv2d(pool2, conv3_w, strides = [1,1,1,1], padding= 'SAME')+conv3_b

relu3= tf.nn.relu(conv3)



print(relu3)
"""flatten layer"""

flatten = tf.reshape(relu3, [-1, 8*8*10])



"""first fully connect layer"""

fc1 = tf.layers.dense(inputs= flatten, units = 512, activation = tf.nn.relu, use_bias =True)



"""first fully connect layer"""

fc2 = tf.layers.dense(inputs= fc1, units = 512, activation = tf.nn.relu, use_bias =True)



"""output layer"""

output = tf.layers.dense(inputs=fc2, units = Num_Classes, activation = None, use_bias= True)



"""loss function"""

loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=output))



"""accuracy function"""

accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis = 1), Y[:,0]), dtype = tf.float32))



"""optimizer"""

opt = tf.train.AdamOptimizer(0.0001).minimize(loss)

"""Initiate the parameters"""

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)
from tqdm import tqdm_notebook as tqdm



"""Training loop"""

EPOCHS = 2

BATCH_SIZE = 64



for epoch in range(0, EPOCHS):

    for step in tqdm(range(int(len(x_train)/BATCH_SIZE)), desc=('Epoch '+str(epoch))):

        """get next batch of training data"""

        x_batch = x_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        y_batch = y_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        """train"""

        loss_value, _ = sess.run([loss, opt], feed_dict={X: x_batch, Y: y_batch})

        

        """her, to make the code simple, I only use the first 1000 images from testing dataset to test the network."""

        loss_value, accuracy_value = sess.run([loss, accu], feed_dict= {X:x_test[:1000], Y: y_test[:1000]})

        print('Epoch loss: ', loss_value,'  accuracy:  ', accuracy_value)
import numpy as np

import matplotlib.pyplot as plt 



"""extract first convolution layer filters"""

conv1_w_extract = sess.run(conv1_w)

print(conv1_w_extract.shape)



plt.figure(figsize = (20,20))



"""show first 10 filters"""

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(np.reshape(conv1_w_extract[:,:,:,i]*100, [3,3,3]))
"""extract first convolution layer feature maps"""

conv1_fmaps = sess.run(relu1, feed_dict = {X: [x_train[0]]})

print(conv1_fmaps.shape)



plt.figure(figsize = (20,20))

"""show first 10 fmaps"""

for i in range(10):

    plt.subplot(3, 10, i+1)

    plt.imshow(np.reshape(conv1_fmaps[0,:,:,i], [32,32]))
from gensim.models import Word2Vec

import gensim



model = gensim.models.KeyedVectors.load_word2vec_format("../input/GoogleNews-vectors-negative300.bin", binary=True, limit = 10000)



print('Google W2V model is loaded')
corpus = ['apple is fruit',

          'banana is fruit',

          'peach is fruit',

          'the color of apple is red',

          'the skin of banana is yellow',

          'peach is pink',

          'fruit has color',

          'monkey like yellow banana',

          'kids like red apple',

          'monkey like pink peach',

          'yellow red and pink are color',

         ]
stop_words = ["is", "the", "of", "and", "are", "like","has"]
def remove_stop_words(corpus):

    results = []

    for text in corpus:

        tmp = text.split(' ')

        for stop_word in stop_words:

            if stop_word in tmp:

                tmp.remove(stop_word)

        results.append(" ".join(tmp))

        

    return results
corpus = remove_stop_words(corpus)
words = []

for text in corpus:

    for word in text.split(' '):

        words.append(word)

        

words = set(words)
words
"""Data Generation"""



word2int = {}



for i,word in enumerate(words):

    word2int[word] = i

    

sentences = []

for sentence in corpus:

    sentences.append(sentence.split())

    

WINDOW_SIZE = 2



data = []

for sentence in sentences:

    for idx, word in enumerate(sentence):

        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence) + 1)]:

            if neighbor !=word:

                data.append([word, neighbor])
import pandas as pd

for text in corpus:

    print(text)



df = pd.DataFrame(data, columns = ['input', 'label'])

df.head(10)
df.shape
word2int
"""Define Tensorflow Graph"""



import tensorflow as tf

import numpy as np



ONE_HOT_DIM = len(words)



# function to convert numbers to one hot vectors

def to_one_hot_encoding(data_point_index):

    one_hot_encoding = np.zeros(ONE_HOT_DIM)

    one_hot_encoding[data_point_index] = 1

    return one_hot_encoding



X = [] # input word

Y = [] # target word



for x, y in zip(df['input'], df['label']):

    X.append(to_one_hot_encoding(word2int[ x ]))

    Y.append(to_one_hot_encoding(word2int[ y ]))



# convert them to numpy arrays

X_train = np.asarray(X)

Y_train = np.asarray(Y)



# making placeholders for X_train and Y_train

x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))



# word embedding will be 2 dimension for 2d visualization

EMBEDDING_DIM = 2 



# hidden layer: which represents word vector eventually

W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))

b1 = tf.Variable(tf.random_normal([1])) #bias

hidden_layer = tf.add(tf.matmul(x,W1), b1)



# output layer

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))

b2 = tf.Variable(tf.random_normal([1]))

prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))



# loss function: cross entropy

loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))



# training operation

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
"""Training"""



sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init) 



iteration = 20000

for i in range(iteration):

    # input is X_train which is one hot encoded word

    # label is Y_train which is one hot encoded neighbor word

    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})

    if i % 3000 == 0:

        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))
# Now the hidden layer (W1 + b1) is actually the word look up table

vectors = sess.run(W1 + b1)

print(vectors)
"""Word Vector in Table"""



w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])

w2v_df['word'] = words

w2v_df = w2v_df[['word', 'x1', 'x2']]

w2v_df

"""Word Vector in 2D Chart"""



import matplotlib.pyplot as plt



fig, ax = plt.subplots()



for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):

    ax.annotate(word, (x1,x2 ))

    

PADDING = 1.0

x_axis_min = np.amin(vectors, axis=0)[0] - PADDING

y_axis_min = np.amin(vectors, axis=0)[1] - PADDING

x_axis_max = np.amax(vectors, axis=0)[0] + PADDING

y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

 

plt.xlim(x_axis_min,x_axis_max)

plt.ylim(y_axis_min,y_axis_max)

plt.rcParams["figure.figsize"] = (20,20)



plt.show()