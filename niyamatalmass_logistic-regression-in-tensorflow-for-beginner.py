# pandas for handling our data

import pandas as pd

# matplotlib and seaborn for visualize our data

import matplotlib.pyplot as plt

import seaborn as sns

# numpy for numeric operations

import numpy as np

# tensorflow! our machine learning library

import tensorflow as tf

# train_test_split from sklearn for splitting our data into train and test set

from sklearn.model_selection import train_test_split

# OneHotEncoder from sklearn for converting features and labels to one-hot encoding

from sklearn.preprocessing import OneHotEncoder

#%matplotlib inline sets the backend of matplotlib to the 'inline' backend. 

%matplotlib inline
# load the data

df = pd.read_csv('../input/Iris.csv')

# print some of data

df.head()
# how many data in each species

df['Species'].value_counts()
df.describe()
# let's visualize the data with sepalLength and sepalWidth 

sns.FacetGrid(df, hue='Species', size=5).map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm').add_legend()
# let's drop Id column because we don't need it

df = df.drop('Id', axis=1)

# convert Species name to numerical value

# Iris setosa = 1

# Iris versicolor = 2

# Irsi virginica = 3

df['Species'] = df['Species'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1, 2, 3])

# now let's print some of the data

df.head(5)
# X is our features ('SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm')

X = df.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# y is our labels

y = df.loc[:, ['Species']]
# declare OneHotEncoder from sklearn

oneHot = OneHotEncoder()

# fit our X to oneHot encoder 

oneHot.fit(X)

# transform

X = oneHot.transform(X).toarray()

# fit our y to oneHot encoder

oneHot.fit(y)

# transform

y = oneHot.transform(y).toarray()



print("Our features X in one-hot format")

print(X)
# let's split our data into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)

# let's print shape of each train and testing

print("Shape of X_train: ", X_train.shape)

print("Shape of y_train: ", y_train.shape)

print("Shape of X_test: ", X_test.shape)

print("Shape of y_test", y_test.shape)
# hyperparameters

learning_rate = 0.0001

num_epochs = 1500

display_step = 1



# for visualize purpose in tensorboard we use tf.name_scope

with tf.name_scope("Declaring_placeholder"):

    # X is placeholdre for iris features. We will feed data later on

    X = tf.placeholder(tf.float32, [None, 15])

    # y is placeholder for iris labels. We will feed data later on

    y = tf.placeholder(tf.float32, [None, 3])

    

with tf.name_scope("Declaring_variables"):

    # W is our weights. This will update during training time

    W = tf.Variable(tf.zeros([15, 3]))

    # b is our bias. This will also update during training time

    b = tf.Variable(tf.zeros([3]))

    

with tf.name_scope("Declaring_functions"):

    # our prediction function

    y_ = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
with tf.name_scope("calculating_cost"):

    # calculating cost

    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)

with tf.name_scope("declaring_gradient_descent"):

    # optimizer

    # we use gradient descent for our optimizer 

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope("starting_tensorflow_session"):

    with tf.Session() as sess:

        # initialize all variables

        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):

            cost_in_each_epoch = 0

            # let's start training

            _, c = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train})

            cost_in_each_epoch += c

            # you can uncomment next two lines of code for printing cost when training

            #if (epoch+1) % display_step == 0:

                #print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch))

        

        print("Optimization Finished!")



        # Test model

        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

        # Calculate accuracy for 3000 examples

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("Accuracy:", accuracy.eval({X: X_test, y: y_test}))