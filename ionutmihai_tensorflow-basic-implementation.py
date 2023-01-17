from subprocess import check_output

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')

test_challange = pd.read_csv('../input/test.csv')
train.describe()

def parseSex(df):

    df.loc[df['Sex'] == 'male', 'Sex'] = 1

    df.loc[df['Sex'] == 'female', 'Sex'] = 0

    return df

def addFamilySize(df):

    df['FamilySize'] = 0

    df['FamilySize'] = df['Parch'] + df['SibSp']

    return df

def normalize(series):

    mean = series.mean()

    stdev= series.std()

    return (series - mean)/stdev



def preprocess_data(df):

    df = parseSex(df)

    df = addFamilySize(df)

    age_median = df['Age'].median()

    df.Age = df.Age.fillna(age_median)

    df.Age = normalize(df.Age)

    df.FamilySize = normalize(df.FamilySize)

    df.Pclass = normalize(df.Pclass)

    return df



def accuracy(predictions, labels):

    predictions=predictions>0.5;

    return (np.sum(predictions==labels)

          / predictions.shape[0])
train = preprocess_data(train)

train, test = train_test_split(train, test_size = 0.2)

predict_data = preprocess_data(test_challange)

features_list = ['Pclass', 'Sex', 'Age', 'FamilySize']

train_features = np.float64(train[features_list].values)

train_target = np.float64(train['Survived'].values)

train_target=train_target.reshape(train_target.shape[0],1)

test_features = np.float64(test[features_list].values)

test_target = np.float64(test.Survived.values)

test_target=test_target.reshape(test_target.shape[0],1)

predict_features=np.float64(predict_data[features_list].values)



print(train_features.shape)

print(train_target.shape)

print(test_features.shape)

print(test_target.shape)

print(predict_features.shape)
features_t = train_features.T

target_t = train_target.T.reshape(1, train_target.shape[0])



test_features_t = test_features.T

test_target_t = test_target.T.reshape(1, test_target.shape[0])





hidden_nodes_1 = 4

hidden_nodes_2 = 4

hidden_nodes_3 = 1


beta = 0.01

graph = tf.Graph()

with graph.as_default():   

    # Input data.

    # Load the training, validation and test data into constants that are

    # attached to the graph.

    tf_train_dataset = tf.constant(train_features)

    tf_train_labels = tf.constant(train_target)

    tf_valid_dataset = tf.constant(test_features)

    tf_test_dataset = tf.constant(test_features)

    tf_final_dataset = tf.constant(predict_features)

    weights_1 = tf.Variable(tf.truncated_normal([hidden_nodes_1,hidden_nodes_2], dtype=tf.float64))

    biases_1 = tf.Variable(tf.zeros([hidden_nodes_2], dtype=tf.float64))

    weights_2 = tf.Variable(tf.truncated_normal([hidden_nodes_2,hidden_nodes_3], dtype=tf.float64))

    biases_2 = tf.Variable(tf.zeros([hidden_nodes_3], dtype=tf.float64))

    

    weights_3 = tf.Variable(tf.truncated_normal([hidden_nodes_3,1], dtype=tf.float64))

    biases_3 = tf.Variable(tf.zeros([1], dtype=tf.float64))

    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1 #[712,4]*[4,4]=[712,4]

    relu_layer_1= tf.nn.relu(logits_1)

    logits_2 = tf.matmul(relu_layer_1, weights_2) + biases_2 #[712,4]*[4,1]=[712,1]

    relu_layer_2=tf.nn.relu(logits_2)

    logits_3=tf.matmul(relu_layer_2,weights_3) + biases_3 #[712,4]*[1,1]=[712,1]

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_3, labels=tf_train_labels))

    

    # Loss function with L2 Regularization with beta=0.01

    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)+ tf.nn.l2_loss(weights_3)

    loss = tf.reduce_mean(loss + beta * regularizers)



    # Optimizer.

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)



    # Predictions for the training

    train_prediction = tf.nn.sigmoid(logits_3)

    

    # Predictions for validation 

    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1

    relu_layer_1= tf.nn.relu(logits_1)

    logits_2 = tf.matmul(relu_layer_1, weights_2) + biases_2

    relu_layer_2=tf.nn.relu(logits_2)

    logits_3=tf.matmul(relu_layer_2,weights_3) +biases_3

    

    valid_prediction = tf.nn.sigmoid(logits_3)

    

    test_prediction =  valid_prediction

    

     # Predictions for final submision 

    logits_1 = tf.matmul(tf_final_dataset, weights_1) + biases_1

    relu_layer_1= tf.nn.relu(logits_1)

    logits_2 = tf.matmul(relu_layer_1, weights_2) + biases_2

    relu_layer_2=tf.nn.relu(logits_2)

    logits_3=tf.matmul(relu_layer_2,weights_3) +biases_3

    

    final_prediction = tf.nn.sigmoid(logits_3)

    

num_steps = 801

ids = predict_data.index.values

final = pd.DataFrame({

    'PassengerId': ids,

    'Survived': np.zeros(ids.shape[0])

})



with tf.Session(graph=graph) as session:

    # This is a one-time operation which ensures the parameters get initialized as

    # we described in the graph: random weights for the matrix, zeros for the

    # biases. 

    tf.global_variables_initializer().run()

    print('Initialized')

    for step in range(num_steps):

        # Run the computations. We tell .run() that we want to run the optimizer,

        # and get the loss value and the training predictions returned as numpy

        # arrays.

        _, l, predictions = session.run([optimizer, loss, train_prediction])

        if (step % 100 == 0):

            print('Loss at step %d: %f' % (step, l))

            print('Training accuracy: %.1f%%' % accuracy(

            predictions, train_target))

            # Calling .eval() on valid_prediction is basically like calling run(), but

            # just to get that one numpy array. Note that it recomputes all its graph

            # dependencies.

            print('Validation accuracy: %.1f%%' % accuracy(

                train_prediction.eval(), train_target))

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_target))

    final['Survived']=np.int64(final_prediction.eval()>0.5)

final

final.to_csv("titanic_predictions.csv", index=False)