# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas  # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import tensorflow as tf # TensorFlow
#Read train and test Set

titanic_train = pandas.read_csv("../input/train.csv")

titanic_test = pandas.read_csv("../input/test.csv")



# The columns we'll use to predict the target

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

print(titanic_test.describe())
## Preprocess training set

titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())



titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 0

titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 1



titanic_train["Embarked"] = titanic_train["Embarked"].fillna('S')

titanic_train.loc[titanic_train["Embarked"] == "S", "Embarked"] = 0

titanic_train.loc[titanic_train["Embarked"] == "C", "Embarked"] = 1

titanic_train.loc[titanic_train["Embarked"] == "Q", "Embarked"] = 2



## Preprocess test set

titanic_test["Age"] = titanic_test["Age"].fillna(titanic_train["Age"].median())



titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2



titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
batch_size = 50

numberHiddenUnits=2000

num_labels=1

#test_dataset=titanic_test



train_predictors =np.array(titanic_train[predictors].iloc[:,:])

# The target we're using to train the algorithm.

train_labels=np.array(titanic_train["Survived"])



graph = tf.Graph()#Initialize a computational graph

with graph.as_default():



  # Input data. For the training data, we use a placeholder that will be fed

  # at run time with a training minibatch.

  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, train_predictors.shape[1]))

  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

  #tf_valid_dataset = tf.constant(valid_dataset)

  tf_test_dataset = tf.placeholder(tf.float32,shape=(batch_size, train_predictors.shape[1]))

  

  # Variables.

  weights = tf.Variable(tf.truncated_normal([train_predictors.shape[1], numberHiddenUnits]))

  weightsLayer3 = tf.Variable(tf.truncated_normal([numberHiddenUnits, num_labels]))

  biases = tf.Variable(tf.zeros([numberHiddenUnits]))

  

  # Training computation.

  #logits = tf.matmul(tf_train_dataset, weights) + biases

  logits = tf.nn.relu(tf.matmul(tf_train_dataset, weights, transpose_a=False, transpose_b=False) + biases, name=None)

  layer3 = tf.matmul(logits, weightsLayer3, transpose_a=False, transpose_b=False) 



  loss = tf.reduce_mean((layer3, tf_train_labels))

  #print(layer3.get_shape(),tf_train_labels.get_shape())

  

  # Optimizer.

  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  

  # Predictions for the training, validation, and test data.

  train_prediction = layer3



  #previousValPred=tf.nn.relu(tf.matmul(tf_valid_dataset, weights, transpose_a=False, transpose_b=False) + biases, name=None)

  #valid_prediction = tf.nn.softmax(tf.matmul(previousValPred,weightsLayer3, transpose_a=False, transpose_b=False)) 

  previousTestPred=tf.nn.relu(tf.matmul(tf_test_dataset, weights, transpose_a=False, transpose_b=False) + biases, name=None)                               

  test_prediction =tf.nn.softmax(tf.matmul(previousTestPred,weightsLayer3, transpose_a=False, transpose_b=False)) 
num_steps = 1500

train_dataset=(titanic_train[predictors])

with tf.Session(graph=graph) as session:

  init_new_vars_op =tf.initialize_all_variables()

  session.run(init_new_vars_op)

  print("Initialized")

  for step in range(num_steps):

    # Pick an offset within the training data, which has been randomized.

    # Note: we could use better randomization across epochs.

    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

    # Generate a minibatch.

    #print(offset,(offset + batch_size))

    batch_data = train_predictors[offset:(offset + batch_size), :]

    batch_labels = train_labels[offset:(offset + batch_size)]

    # Prepare a dictionary telling the session where to feed the minibatch.

    # The key of the dictionary is the placeholder node of the graph to be fed,

    # and the value is the numpy array to feed to it.

    batch_data = np.reshape(batch_data, (-1, 50))

    #tf_train_dataset = np.reshape(tf_train_dataset, (-1, 50))

    batch_labels = np.reshape(batch_labels, (-1, 50))

    

    print(train_prediction.get_shape(),tf_train_dataset.get_shape())

    

    feed_dict = {tf_train_dataset : batch_data.transpose(), tf_train_labels : batch_labels.transpose()}

    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

    if (step % 500 == 0):

      print("Minibatch loss at step %d: %f" % (step, l))

      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

      print("Validation accuracy: %.1f%%" % accuracy(

        valid_prediction.eval(), valid_labels))

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
np.array(titanic_train["Survived"])[0:6]