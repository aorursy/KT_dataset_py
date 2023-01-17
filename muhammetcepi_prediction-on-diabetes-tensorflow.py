# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import time

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_diabetes = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
X = np.asarray(df_diabetes[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]])

X[0:5]

y = np.asarray(df_diabetes['Outcome'])

y= pd.get_dummies(y).values

y [0:5]
X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

# numFeatures is the number of features in our input data.

# In the iris dataset, this number is '4'.

numFeatures = X_train.shape[1]



# numLabels is the number of classes our data points can be in.

# In the iris dataset, this number is '3'.

numLabels = y_train.shape[1]





# Placeholders

# 'None' means TensorFlow shouldn't expect a fixed number in that dimension

X = tf.placeholder(tf.float32, [None, numFeatures]) # Iris has 4 features, so X is a tensor to hold our data.

yGold = tf.placeholder(tf.float32, [None, numLabels]) # This will be our correct answers matrix for 3 classes.
W = tf.Variable(tf.zeros([8, 2]))  # 4-dimensional input and  3 classes

b = tf.Variable(tf.zeros([2])) # 3-dimensional output [0,0,1],[0,1,0],[1,0,0]
#Randomly sample from a normal distribution with standard deviation .01



weights = tf.Variable(tf.random_normal([numFeatures,numLabels],

                                       mean=0,

                                       stddev=0.01,

                                       name="weights"))



bias = tf.Variable(tf.random_normal([1,numLabels],

                                    mean=0,

                                    stddev=0.01,

                                    name="bias"))
# Three-component breakdown of the Logistic Regression equation.

# Note that these feed into each other.

apply_weights_OP = tf.matmul(X, weights, name="apply_weights")

add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 

activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
# Number of Epochs in our training

numEpochs = 2000



# Defining our learning rate iterations (decay)

learningRate = tf.train.exponential_decay(learning_rate=0.0008,

                                          global_step= 1,

                                          decay_steps=X_train.shape[0],

                                          decay_rate= 0.95,

                                          staircase=True)
#Defining our cost function - Squared Mean Error

cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")



#Defining our Gradient Descent

training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)
# Create a tensorflow session

sess = tf.Session()



# Initialize our weights and biases variables.

init_OP = tf.global_variables_initializer()



# Initialize all tensorflow variables

sess.run(init_OP)

# argmax(activation_OP, 1) returns the label with the most probability

# argmax(yGold, 1) is the correct label

correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))



# If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy

accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))



# Summary op for regression output

activation_summary_OP = tf.summary.histogram("output", activation_OP)



# Summary op for accuracy

accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)



# Summary op for cost

cost_summary_OP = tf.summary.scalar("cost", cost_OP)



# Summary ops to check how variables (W, b) are updating after each iteration

weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))

biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))



# Merge all summaries

merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])



# Summary writer

writer = tf.summary.FileWriter("summary_logs", sess.graph)
# Initialize reporting variables

cost = 0

diff = 1

epoch_values = []

accuracy_values = []

cost_values = []

# Training epochs

for i in range(numEpochs):

    if i > 1 and diff < .0001:

        print("change in cost %g; convergence."%diff)

        break

    else: 

        # Run training step

        step = sess.run(training_OP, feed_dict={X: X_train, yGold: y_train})

        # Report occasional stats

        if i % 10 == 0:

            # Add epoch to epoch_values

            epoch_values.append(i)

            # Generate accuracy stats on test data

            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: X_train, yGold: y_train})

            # Add accuracy to live graphing variable

            accuracy_values.append(train_accuracy)

            # Add cost to live graphing variable

            cost_values.append(newCost)

            # Re-assign values for variables

            diff = abs(newCost - cost)

            cost = newCost



            #generate print statements

            print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))





# How well do we perform on held-out test data?

print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 

                                                     feed_dict={X:X_test , 

                                                                yGold: y_test})))