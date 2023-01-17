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
import tensorflow as tf
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

print(test.shape)
train.head()
X_train = train.drop('label', axis=1).values

Y_train = train['label'].values
def to_categorical(y, num_classes=None):

    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments

        y: class vector to be converted into a matrix

            (integers from 0 to num_classes).

        num_classes: total number of classes.

    # Returns

        A binary matrix representation of the input.

    """

    y = np.array(y, dtype='int').ravel()

    if not num_classes:

        num_classes = np.max(y) + 1

    n = y.shape[0]

    categorical = np.zeros((n, num_classes))

    categorical[np.arange(n), y] = 1

    return categorical



Y_train = to_categorical(Y_train,10)
Y_train[1]
learning_rate = 0.001

n_steps = 501

display_step = 100

batch_size = 200

num_classes = 10
# variables and placeholders

X = tf.placeholder(tf.float32,[None,X_train.shape[1]])

Y_ = tf.placeholder(tf.float32, [None,num_classes])



W1 = tf.Variable(tf.truncated_normal([X_train.shape[1],num_classes],stddev=0.1))

b1 = tf.Variable(tf.zeros([num_classes]))



# model

Ylogits = tf.matmul(X,W1) + b1

Y = tf.nn.softmax(Ylogits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_,logits=Ylogits))



train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_pred = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))

acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
for i in range(n_steps):

    k = 0

    n_digits = X_train.shape[0]

    for j in range(0,n_digits,batch_size):

        trainData = {X:X_train[j:j+batch_size], Y_:Y_train[j:j+batch_size]}

#         print(trainData)

        sess.run(train_step, feed_dict=trainData)

     

    if i % display_step == 0:

        a,l = sess.run([acc,loss], feed_dict=trainData)

        print("%s Step - Acc: %s, Loss: %s"%(i,a,l))
testData = {X:test}

pred_result = sess.run(Y,feed_dict=testData)

result = sess.run(tf.argmax(pred_result,1))
sub = pd.DataFrame({'Label': result}, index=range(1,len(result)+1))

sub.to_csv('submission_single_nn.csv',index_label='ImageId')