import pandas as pd

import numpy as np

import tensorflow as tf

import random as rnd

import matplotlib.pyplot as plt

import math

from sklearn.model_selection import train_test_split
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

print(len(train_data))

print(len(test_data))
train_data.head()
test_data.head()
train_dropped = train_data.drop(['Ticket', 'Cabin', 'Name'], axis=1)

test_dropped = test_data.drop(['Ticket', 'Cabin', 'Name'], axis=1)

train_dropped.head()

test_dropped.head()
# Get dummy variables

dummies_sex_train = pd.get_dummies(train_dropped['Sex'])

dummies_embarked_train = pd.get_dummies(train_dropped['Embarked'])

dummies_sex_test = pd.get_dummies(test_dropped['Sex'])

dummies_embarked_test = pd.get_dummies(test_dropped['Embarked'])



# Insert dummy variables

train_new = pd.concat([train_dropped, dummies_sex_train, dummies_embarked_train], axis=1)

test_new = pd.concat([test_dropped, dummies_sex_test, dummies_embarked_test], axis=1)



# Drop old variables

train_new = train_new.drop(['PassengerId', 'Sex', 'Embarked'], axis=1)

test_new = test_new.drop(['PassengerId', 'Sex', 'Embarked'], axis=1)



# Fill the NaN with zeros

train_new = train_new.fillna(0)

test_new = test_new.fillna(0)

train_new.head(15)
test_new.head(10)
# Split the training data randomly to 20% cross validation set

train_final, crossvalid_final = train_test_split(train_new, test_size = 0.2)

num_train = len(train_final)

num_cv = len(crossvalid_final)



#Check the training set

train_final.head()
# Check the number of two sets

print(num_train)

print(num_cv)
# check the cross validation set

crossvalid_final.head()
# Split the data to features and labels

train_features = train_final.ix[:, 1:]

train_label = train_final.loc[:,['Survived']]

cv_features = crossvalid_final.ix[:, 1:]

cv_labels = crossvalid_final.loc[:,['Survived']]
# Check features

train_features.head()
len_feature = 10

x = tf.placeholder(tf.float32, [None, len_feature])

y = tf.placeholder(tf.float32, [None, 1])

# Neural Network Hyperparameters

hidden_layer1_nodes = 64

hidden_layer2_nodes = 32

output_nodes = 1

batch_size = 64

learning_rate = 0.0001



training_loss = []

validation_loss = []
def neural_network(x_tensor):



    # Initialize weights and biases for hidden layer

    # Layer 1

    hidden_weights1 = tf.Variable(tf.truncated_normal([len_feature, hidden_layer1_nodes], stddev=0.1))

    hidden_biases1 = tf.Variable(tf.zeros(hidden_layer1_nodes))

    

    # Layer 2

    hidden_weights2 = tf.Variable(tf.truncated_normal([hidden_layer1_nodes, hidden_layer2_nodes], stddev=0.1))

    hidden_biases2 = tf.Variable(tf.zeros(hidden_layer2_nodes))



    # Output layer

    output_weights = tf.Variable(tf.truncated_normal([hidden_layer2_nodes, output_nodes], stddev=0.1))

    output_biases = tf.Variable(tf.zeros(output_nodes))



    # Build the structure of neural network

    x_tensor = tf.add(tf.matmul(x_tensor, hidden_weights1), hidden_biases1)

    x_tensor = tf.nn.relu(x_tensor)

    x_tensor = tf.add(tf.matmul(x_tensor, hidden_weights2), hidden_biases2)

    x_tensor = tf.nn.relu(x_tensor)

    x_tensor = tf.add(tf.matmul(x_tensor, output_weights), output_biases)

    

    # Sigmoid output

    x_tensor = tf.sigmoid(x_tensor)



    return x_tensor
def get_batches(data, batch_size):

    

    idx = data.index.tolist()  # get all possible indexes

    np.random.shuffle(idx)     # shuffle indexes

    

    features = []

    labels = []

    data_size = len(data)

    for i in range(0, int(len(data)/batch_size)):

        batch_idx = idx[i*batch_size:min(data_size-1, (i+1)*batch_size-1)]

        batch = data.ix[batch_idx]       # get the batch

        features.append(batch.ix[:, 1:]) # get features from batch

        labels.append(batch.loc[:,['Survived']]) # get label from batch

    

    return zip(features, labels)
def train():

    logits = neural_network(x)    # get model output

    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits, y))))    # calculate cost

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)    # Use Adam optimizer

    correct_pred = tf.equal(logits>0.5, y>0.5)    # Collect correct predictions

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')    # calculate accuracy

    

    #Number of epochs

    num_epochs = 500

    

    with tf.Session() as sess:

        

        # Initialization

        sess.run(tf.global_variables_initializer())

        

        # Each epoch

        for epoch in range(num_epochs):  

            

            batch_idx = 0

            

            # Go through each batches

            for feature_batch, label_batch in get_batches(train_final, batch_size):

                

                batch_idx += 1

                

                # Train the network

                sess.run(optimizer, feed_dict={x:feature_batch, y:label_batch})

                

                # Calculate loss

                train_loss = sess.run(cost, feed_dict={x:feature_batch, y:label_batch})

                cv_loss = sess.run(cost, feed_dict={x:cv_features, y:cv_labels})

                training_loss.append(train_loss)

                validation_loss.append(cv_loss)

                

                # Calculate cross validation accuracy

                validation_accuracy = sess.run(accuracy, feed_dict={x:cv_features, y:cv_labels})

                

                # Print out results

                # print('Epoch {:>2}, Titanic Batch {}:  '.format(epoch + 1, batch_idx), end='')

                # print('Current loss is : %f.    Validation accuracy is : %.2f%%'%(train_loss,validation_accuracy*100))

                

        prediction = sess.run(logits, feed_dict={x:test_new})

    return prediction
prediction = train()
plt.figure(1)

plt.plot(training_loss,label='training_loss')

plt.plot(validation_loss,label='validation_loss')

plt.legend()

plt.show()
result = []

for pre in prediction:

    pre = math.ceil(pre-0.5)

    result.append(pre)

print(result)