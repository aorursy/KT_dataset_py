# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



import tensorflow as tf





import os



# to make this notebook's output stable across runs

def reset_graph(seed=42):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)



# To plot pretty figures



import matplotlib



plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12



# Where to save the figures

PROJECT_ROOT_DIR = "."

CHAPTER_ID = "cnn"



def save_fig(fig_id, tight_layout=True):

    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format='png', dpi=300)

    

def plot_image(image):

    plt.imshow(image, cmap="gray", interpolation="nearest")

    plt.axis("off")



def plot_color_image(image):

    plt.imshow(image.astype(np.uint8),interpolation="nearest")

    plt.axis("off")

    
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)
X_train = train.iloc[:,1:].values.astype(np.float32)

X_topred = test.values.astype(np.float32)

y_train = train.iloc[:,0].values.astype(np.int32).ravel()

train_size = X_train.shape[0]

train_size
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

root_logdir = "tf_logs"

logdir = "{}/run-{}/".format(root_logdir, now)

height = 28

width = 28

channels =1

n_inputs = height * width



conv1_fmaps = 32

conv1_ksize = 3

conv1_stride = 1

conv1_pad = "SAME"



conv2_fmaps = 64

conv2_ksize = 3

conv2_stride = 2

conv2_pad = "SAME"



pool3_fmaps = conv2_fmaps



n_fc1 = 54

n_outputs = 10



reset_graph()



with tf.name_scope("inputs"):

    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")

    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])

    y = tf.placeholder(tf.int32, shape=[None], name="y")

    

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps,

                         kernel_size=conv1_ksize,

                         strides = conv1_stride,

                         padding=conv1_pad,

                         activation=tf.nn.relu,

                         name="conv1")



conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps,

                         kernel_size=conv2_ksize,

                         strides = conv2_stride,

                         padding=conv2_pad,

                         activation=tf.nn.relu,

                         name="conv2")



with tf.name_scope("pool3"):

    pool3 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps*7*7])



with tf.name_scope("fc1"):

    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

    

with tf.name_scope("output"):

    logits = tf.layers.dense(fc1, n_outputs, name="output")

    Y_proba = tf.nn.softmax(logits, name="Y_proba" )

    

with tf.name_scope("train"):

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)

    loss = tf.reduce_mean(xentropy)

    optimizer = tf.train.AdamOptimizer()

    training_op = optimizer.minimize(loss)

    

with tf.name_scope("eval"):

    correct = tf.nn.in_top_k(logits, y, 1)

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    

with tf.name_scope("init_and_save"):

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    

predict = tf.argmax(Y_proba,1)



file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())



    
n_epochs = 10

batch_size = 100

n_batches = (train_size-2000) // batch_size

print(n_batches)



def next_batch(X, y, epoch, iteration, batch_size):

    np.random.seed(epoch*n_batches + iteration)

    indices = np.random.randint(42000, size=batch_size)

    X_batch = X[indices]

    y_batch = y[indices]

    return X_batch, y_batch

X_test = X_train[40000:41999]

y_test = y_train[40000:41999]

    
with tf.Session() as sess:

    init.run()

    for epoch in range(n_epochs):

        for iteration in range(n_batches):

            X_batch, y_batch = next_batch(X_train, y_train, epoch, iteration, batch_size)

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

        if iteration%40 ==0: print(".", end="")

        print("\n", epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)



        save_path = saver.save(sess, "./CNN-vanilla.ckpt")

file_writer.close()
test = pd.read_csv('../input/test.csv')

X_test = test.values

predicted_lables = np.zeros(28000)

with tf.Session() as sess:

    saver.restore(sess,'./CNN-vanilla.ckpt')

#     y_pred = tf.arg_max(logits.eval(feed_dict={X:X_test}), dimension=1)

    for i in range(0,28000//100):

        predicted_lables[i*100 : (i+1)*100] = predict.eval(feed_dict={X: X_test[i*100 : (i+1)*100]} )

        print(".", end="")
df = pd.DataFrame({

    'ImageId': pd.Series(range(1,len(predicted_lables)+1),index=list(range(len(predicted_lables))),dtype='int32'),

    'Label' : pd.Series(predicted_lables, dtype='int32')

})

df



df.to_csv("cnn-vanilla1.csv",index=False)