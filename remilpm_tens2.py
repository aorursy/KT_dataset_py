#Multiple logistics regression using Google Tensor flow

from __future__ import print_function

import math

from IPython import display

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from sklearn import metrics

import tensorflow as tf

from tensorflow.python.data import Dataset

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import time

tf.logging.set_verbosity(tf.logging.ERROR)

pd.options.display.max_rows = 10

pd.options.display.float_format = '{:.1f}'.format

import seaborn as sns

import matplotlib.pyplot as plt

print(os.listdir("../input"))

#Read the file

DB1=pd.read_csv("../input/diabetes.csv")

DB1.head()
#Check for null values

DB1.isnull().sum()
#Correlation states how the features are related to each other or the target variable.

#Correlation can be positive ie, increase in one value of feature increases the value of the target variable or

#negative ie,increase in one value of feature decreases the value of the target variable

#Heatmap makes it easy to identify which features are most related to the target variable.

#get correlations of each features in dataset

corr = DB1.corr()

top_corr_features = corr.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(DB1[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Verdict : Glucose, BMI and Age are having high correlation with diabetes
#Randomizing the data to make sure that no pathological ordering effects  the performance of Stochastic Gradient Descent.

DB2 = DB1.reindex(np.random.permutation(DB1.index))

DB2.head()
#Examine the data

DB2.describe()
#Get the size of the data

DB2.shape
#Split the data in to train and validate

DB3_Train=DB2.head(468)

DB3_Val=DB2.tail(300)
# Define the input feature

Diabetic_Feature_Train=DB3_Train[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]

Diabetic_Feature_Val=DB3_Val[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]



Diabetic_Feature_Train.head()
Diabetic_Feature_Val.head()
# Define the label.

Diabetic_Targets_Train = DB3_Train["Outcome"]

Diabetic_Targets_Val= DB3_Val["Outcome"]
Diabetic_Targets_Train.head()
Diabetic_Targets_Val.head()
Diabetic_Feature_Train.shape,Diabetic_Targets_Train.shape,Diabetic_Feature_Val.shape,Diabetic_Targets_Val.shape
#Perform logistics regression to compare with tensor flow

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(Diabetic_Feature_Train, Diabetic_Targets_Train)
y_pred = logreg.predict(Diabetic_Feature_Val)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(Diabetic_Feature_Val, Diabetic_Targets_Val)))
#Now  start preparing data for logistics regression for tensor flow

X=DB2.drop(labels=['Outcome'], axis=1).values

y=DB2.Outcome.values

X.shape,y.shape

# set seed for numpy and tensorflow

# set for reproducible results

seed = 5

np.random.seed(seed)

tf.set_random_seed(seed)
# set replace=False, Avoid double sampling

train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
# diff set

test_index = np.array(list(set(range(len(X))) - set(train_index)))

train_X = X[train_index]

train_y = y[train_index]

test_X = X[test_index]

test_y = y[test_index]
# Begin building the model framework

# Declare the variables that need to be learned and initialization

# There are 8 features here, A's dimension is (8, 1)

A = tf.Variable(tf.random_normal(shape=[8, 1]))

b = tf.Variable(tf.random_normal(shape=[1, 1]))

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
# Define placeholders

data = tf.placeholder(dtype=tf.float32, shape=[None, 8])

target = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# Declare the model you need to learn

mod = tf.matmul(data, A) + b
# Declare loss function

# Use the sigmoid cross-entropy loss function,

# first doing a sigmoid on the model result and then using the cross-entropy loss function

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))
# Define the learning rate， batch_size etc.

learning_rate = 0.003

batch_size = 30

iter_num = 1500
# Define the optimizer

opt = tf.train.GradientDescentOptimizer(learning_rate)
# Define the goal

goal = opt.minimize(loss)
# Define the accuracy

# The default threshold is 0.5, rounded off directly

prediction = tf.round(tf.sigmoid(mod))

# Bool into float32 type

correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)

# Average

accuracy = tf.reduce_mean(correct)

# End of the definition of the model framework
# Start training model

# Define the variable that stores the result

loss_trace = []

train_acc = []

test_acc = []
# training model

for epoch in range(iter_num):

    # Generate random batch index

    batch_index = np.random.choice(len(train_X), size=batch_size)

    batch_train_X = train_X[batch_index]

    batch_train_y = np.matrix(train_y[batch_index]).T

    sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})

    temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})

    # convert into a matrix, and the shape of the placeholder to correspond

    temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: np.matrix(train_y).T})

    temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: np.matrix(test_y).T})

    # recode the result

    loss_trace.append(temp_loss)

    train_acc.append(temp_train_acc)

    test_acc.append(temp_test_acc)

    # output

    if (epoch + 1) % 300 == 0:

        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,

                                                                          temp_train_acc, temp_test_acc))
# Visualization of the results

# loss function

plt.plot(loss_trace)

plt.title('Cross Entropy Loss')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
# accuracy

plt.plot(train_acc, 'b-', label='train accuracy')

plt.plot(test_acc, 'k-', label='test accuracy')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.title('Train and Test Accuracy')

plt.legend(loc='best')

plt.show()