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
mushrooms = pd.read_csv("../input/mushrooms.csv")

mushrooms.head()
from sklearn.feature_extraction import DictVectorizer



def encode_onehot(df):

    vec = DictVectorizer()

    

    vec_data = pd.DataFrame(vec.fit_transform(df.to_dict(orient='records')).toarray())

    vec_data.columns = vec.get_feature_names()

    vec_data.index = df.index

    return vec_data



mushrooms = encode_onehot(mushrooms)

mushrooms.head()
from sklearn.model_selection import train_test_split # helper method to split dataset

train, test = train_test_split(mushrooms, test_size=0.2)
train_y = train[["class=e", "class=p"]] 

train_x = train.drop(["class=e","class=p"], 1)
test_y = test[["class=e", "class=p"]]

test_x = test.drop(["class=e", "class=p"], 1)
import tensorflow as tf

sess = tf.InteractiveSession()



x = tf.placeholder(tf.float32, shape=[None, 117]) # will hold features through feed_dict. Shape is [None, 117]

                                                  # because we have an undefined number of rows and 117 features

y_ = tf.placeholder(tf.float32, shape=[None, 2]) # will hold labels through feed_dict. Shape is [None, 2]

                                                 # beacause we have an undefined number of rows and 2 output classes

                                                 # or labels



W = tf.Variable(tf.zeros([117, 2])) # initialize weights and

b = tf.Variable(tf.zeros([2])) # biases



sess.run(tf.global_variables_initializer())



y = tf.sigmoid(tf.matmul(x, W) + b) # we compute our prediction and use a sigmoid activation function to get

                                    # results as probabilities



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) # cross entropy as our loss function

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # minimize loss with Gradient Descent



correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # used to calculate accuracy later



for i in range(1000):

    train_step.run(feed_dict={x: train_x, y_: train_y})

    print("Training iteration " + str(i) + ": " + str(accuracy.eval(feed_dict={x: train_x, y_: train_y})))



print("Accuracy in test set: ", accuracy.eval(feed_dict={x: test_x, y_: test_y}))
def are_poisonous(mushrooms):

    predictions = sess.run(y, feed_dict={x: mushrooms})

    return [prediction[1] > prediction[0] for prediction in predictions]
mushrooms = mushrooms.sample(5)
mushrooms[["class=e", "class=p"]]
are_poisonous(mushrooms.drop(["class=e","class=p"], 1))