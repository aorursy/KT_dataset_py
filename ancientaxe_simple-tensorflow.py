# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv( '../input/train.csv' )

data = df.as_matrix()
from sklearn.model_selection import train_test_split



train_x, test_x, train_y, test_y = train_test_split( data[:,1:].astype( 'float32' ), 

                                                    data[:,0].astype( 'float32' ), 

                                                    test_size=0.33, 

                                                    random_state=4 )
plt.figure( figsize=( 14,10 ) )

for i in range( 50 ):

    sample = np.random.randint( 0, train_x.shape[0], 1 )[0]

    plt.subplot(5, 10, i+1)

    plt.tick_params( axis='both', which='both', bottom='off', left='off', labelbottom='off', labelleft='off' )

    plt.imshow( train_x[sample].reshape((28, 28)), cmap='gray' )

    plt.title( int( train_y[sample] ) )
print("Min: %.2f" % np.min(train_x))

print("Max: %.2f" % np.max(train_x))

train_x /= 255

test_x /= 255

print("Min: %.2f" % np.min(train_x))

print("Max: %.2f" % np.max(train_x))
train_y = pd.get_dummies( train_y ).as_matrix()

test_y = pd.get_dummies( test_y ).as_matrix()

train_y
import tensorflow as tf
x = tf.placeholder( tf.float32, [None, 784] )



W = tf.Variable( tf.random_normal( [784, 10], mean=0.1, seed=4 ) )

b = tf.Variable( tf.random_normal( [10], mean=0.1, seed=4 ))



# dropout = tf.nn.dropout( tf.matmul( x, W ) + b, 0.75 )

y = tf.nn.softmax( tf.matmul( x, W ) + b )
y_ = tf.placeholder( tf.float32, [None, 10] )
# loss function

cross_entropy = tf.reduce_mean( -tf.reduce_sum( y_ * tf.log( y ), reduction_indices=[1] ) )
# optimise loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

train_size = train_x.shape[0]

train_size
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# training

train_size = train_x.shape[0]

batch_size = 100



sess = tf.InteractiveSession()



tf.global_variables_initializer().run()

for i in range( 0, 500000, batch_size ):

    idx = i%train_size

    sess.run( train_step, feed_dict={x: train_x[idx:idx+batch_size], 

                                     y_: train_y[idx:idx+batch_size]} )

    if i%10000==0:

        loss, acc = sess.run( [cross_entropy, accuracy], feed_dict={x: train_x[idx:idx+batch_size], y_: train_y[idx:idx+batch_size]} )

        print( 'Step: ' + str(i) + ' loss: ' + str(loss) + ' accuracy: ' + str(acc) )
tested_y = sess.run( tf.argmax( y, 1 ), feed_dict = {x: test_x} )

correct_prediction = np.equal( tested_y, np.argmax( test_y, 1 ) )
#correct_prediction = tf.equal( tf.argmax( y,1 ), tf.argmax (y_,1 ) )

p = tf.placeholder(tf.bool, [None,])

accuracy = tf.reduce_mean( tf.cast( p, tf.float32 ) )



print( 'Accuracy (on unseen data):', sess.run( accuracy, feed_dict={ p: correct_prediction } ) )

print('Correct:',

      sum(correct_prediction.astype(int)), 

      ' out of', 

      len(test_x) )
test_combine = np.concatenate( ( np.array( [correct_prediction] ).T,np.array([tested_y]).T,test_x ),axis=1 )

failures = np.array( list( x for x in test_combine if x[0]==0 ) )

print( 'Number of failures: %s'%failures.shape[0] )
plt.figure( figsize=( 14,10 ) )

for i in range( 50 ):

    sample = np.random.randint( 0, failures.shape[0], 1 )[0]

    plt.subplot(5, 10, i+1)

    plt.tick_params( axis='both', which='both', bottom='off', left='off', labelbottom='off', labelleft='off' )

    plt.imshow( failures[sample,2:].reshape((28, 28)), cmap='gray' )

    plt.title( int( failures[sample, 1] ) )
df = pd.read_csv( '../input/test.csv' )

predict_x = df.as_matrix() / 255



predict_y = tf.argmax( y, 1 )

predicted_y = sess.run( predict_y, feed_dict = {x: predict_x} )
plt.figure( figsize=( 14,10 ) )

for i in range( 50 ):

    sample = np.random.randint( 0, predict_x.shape[0], 1 )[0]

    plt.subplot(5, 10, i+1)

    plt.tick_params( axis='both', which='both', bottom='off', left='off', labelbottom='off', labelleft='off' )

    plt.imshow( predict_x[sample].reshape((28, 28)), cmap='gray' )

    plt.title( predicted_y[sample] )
# ImageId,Label

# pd.DataFrame({"ImageId": list(range(1,len(predicted_y)+1)), "Label": predicted_y}).to_csv('output.csv', index=False, header=True)