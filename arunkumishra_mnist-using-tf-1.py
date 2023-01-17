import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



%matplotlib inline
# Load the data

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.head()
train_data.shape
y_data = train_data['label']

y_data = pd.get_dummies(y_data)

X_data = train_data.drop('label',axis=1)

del train_data

plt.imshow(X_data.iloc[3].values.reshape(28,28), cmap='gist_gray')
#Normalize

X_data = X_data / 255.0

test_data = test_data / 255.0
x = tf.placeholder(tf.float32,shape=[None,784])

W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))
# Create the Graph

y = tf.matmul(x,W) + b 
y_true = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
# create training and validation data

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.25, random_state=42)
epoch_accuracies = []

output = pd.read_csv("../input/sample_submission.csv")



# Model

with tf.Session() as sess:

    sess.run(init)

    start, end = 0, 0

    for epochs in range(100):

        for step in range(63):

            start = step*500

            end = (1+step)*500

            batch_x , batch_y = X_train.iloc[start:end,], y_train.iloc[start:end,]

            

            sess.run(train,feed_dict={x:batch_x,y_true:batch_y})



        # Test the Train Model

        matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))

        print('.',end=" ")

        acc = tf.reduce_mean(tf.cast(matches,tf.float32))

        epoch_accuracy = sess.run(acc,feed_dict={x:X_valid,y_true:y_valid})

        epoch_accuracies.append(epoch_accuracy) 

    print ("Train Accuracy:", acc.eval({x: X_train, y_true: y_train}))



    

    # Plot the accuracy

    plt.plot(np.squeeze(epoch_accuracies))

    plt.ylabel('accuracy')

    plt.xlabel('iterations')

    plt.show()

    

    # Make predictions on test data

    prediction = sess.run(y, feed_dict={x: test_data})

    output['Label'] = tf.argmax(prediction,1).eval()
output.to_csv('submission.csv',index=False)