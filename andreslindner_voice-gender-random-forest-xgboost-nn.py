# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/voice.csv')
data.head(10)
data.describe()
data.info()
correlation = data.corr()

correlation
# Plotting correlation matrix

plt.figure(figsize=(15,15))

sns.heatmap(correlation, square=True)
# Importing sklearn libraries

from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn import metrics
X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values
print(y)
# Encoding label (male=1 and female=0)

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print(y)
# Standarizing features

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
# 70-30% of train and test

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.30)
Xtrain, Xtest, ytrain, ytest = np.array(Xtrain,dtype='float32'), np.array(Xtest,dtype='float32'),np.array(ytrain,dtype='float32'),np.array(ytest,dtype='float32')
# Random Forest

random_forest = RandomForestClassifier()

random_forest.fit(Xtrain, ytrain)

y_predicted = random_forest.predict(Xtest)
# Test Accuracy (RF)

print(metrics.accuracy_score(ytest, y_predicted))
# This script shows you how to make a submission using a few

# useful Python libraries.

# It gets a public leaderboard score of 0.76077.

# Maybe you can tweak it and do better...?



import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import numpy as np





# xgboost model

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(Xtrain, ytrain)

y_pred = gbm.predict(Xtest)



# Test Accuracy (xgboost)

print(metrics.accuracy_score(ytest, y_pred))
def convertToOneHot(vector, num_classes=None):

    """

    Converts an input 1-D vector of integers into an output

    2-D array of one-hot vectors, where an i'th input value

    of j will set a '1' in the i'th row, j'th column of the

    output array.



    Example:

        v = np.array((1, 0, 4))

        one_hot_v = convertToOneHot(v)

        print one_hot_v



        [[0 1 0 0 0]

         [1 0 0 0 0]

         [0 0 0 0 1]]

    """



    assert isinstance(vector, np.ndarray)

    assert len(vector) > 0



    if num_classes is None:

        num_classes = np.max(vector)+1

    else:

        assert num_classes > 0

        assert num_classes >= np.max(vector)



    result = np.zeros(shape=(len(vector), num_classes))

    result[np.arange(len(vector)), vector] = 1

    return result.astype(int)

from sklearn.preprocessing import LabelEncoder



# Converting label into one-hot vector

ytrain = LabelEncoder().fit_transform(ytrain)

ytrain = convertToOneHot(ytrain, 2)



ytest = LabelEncoder().fit_transform(ytest)

ytest = convertToOneHot(ytest, 2)
import tensorflow as tf



def layer(input, n_input, n_output, name='hidden_layer'):

    W = tf.Variable(tf.truncated_normal([n_input,n_output], stddev=0.1), name='W')

    B = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_output]), name='B')

    return tf.add(tf.matmul(input,W), B)



# TF Graph 

x = tf.placeholder(tf.float32, shape=[None, 20], name="x")

y = tf.placeholder(tf.float32, shape=[None, 2], name="y")



hidden_1 = tf.nn.relu(layer(x, 20, 15, 'hidden_layer_1'))

hidden_2 = tf.nn.relu(layer(hidden_1, 15, 10, 'hidden_layer_2'))

hidden_3 = tf.nn.relu(layer(hidden_2, 10, 5, 'hidden_layer_3'))

output = layer(hidden_3, 5, 2, 'output')



# Calculating loss function (cross-entropy)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y), name='xent')



# Training

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)

train = optimizer.minimize(loss)



# Accuracy

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name="correct_prediction")

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)



num_epochs = 1000

batch_size = 100

train_size = Xtrain.shape[0]



for epoch in range(num_epochs):

    avg_accuracy = 0.0

    total_batches = int(train_size // batch_size)

    for step in range(total_batches):

        offset = (step * batch_size) % train_size

        batch_data = Xtrain[offset:(offset+batch_size),:]

        batch_labels = ytrain[offset:(offset+batch_size),:]

        _, ac = sess.run([train, accuracy], feed_dict={x: batch_data, y: batch_labels})

        avg_accuracy += ac / total_batches

    validation_accuracy = sess.run([accuracy], feed_dict= {x: Xtest, y: ytest})

    if epoch % 50 == 0:

        print("Epoch:{} training_accuracy={}".format(epoch+1,avg_accuracy))

        print("Epoch:{} testing_accuracy={}".format(epoch+1,validation_accuracy))

        

test_accuracy = sess.run([accuracy], feed_dict={x: Xtest, y: ytest})

print("Testing accuracy = {}".format(test_accuracy))



# validation_accuracy, validation_pred = sess.run([accuracy, validation_prediction],

#                                       feed_dict={x: x_validation, y_: y_validation})  

# test_accuracy, test_pred = sess.run([accuracy, validation_prediction],

#                                 feed_dict={x: x_test, y_: y_test})

# print("Finally, validation_accuracy={}, test_accuracy={}".format(validation_accuracy, test_accuracy))
# Kaggle needs the submission to have a certain format;

# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv

# for an example of what it's supposed to look like.

# submission = pd.DataFrame({ 'id': range(len(Xtest)),

#                             'actual': ytest,

#                             'prediction': y_pred })

# submission.to_csv("submission.csv", index=False)