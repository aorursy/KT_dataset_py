import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display, Image

from pandas import get_dummies

from sklearn.cross_validation import train_test_split

# Config the matlotlib backend as plotting inline in IPython

%matplotlib inline
data = pd.read_csv('../input/Iris.csv', index_col = 0)
data.head()
data.shape
# Visualize data with Seaborn

g=sns.pairplot(data, hue="Species", size= 2.5)
cols = data.columns

features = cols[0:4]

labels = cols[4]

print(features)

print(labels)
#Well conditioned data will have zero mean and equal variance

#We get this automattically when we calculate the Z Scores for the data



data_norm = pd.DataFrame(data)



for feature in features:

    data[feature] = (data[feature] - data[feature].mean())/data[feature].std()



#Show that should now have zero mean

print("Averages")

print(data.mean())



print("\n Deviations")

#Show that we have equal variance

print(pow(data.std(),2))
#Shuffle The data

indices = data_norm.index.tolist()

indices = np.array(indices)

np.random.shuffle(indices)

X = data_norm.reindex(indices)[features]

y = data_norm.reindex(indices)[labels]
# One Hot Encode as a dataframe

y = get_dummies(y)



# Generate Training and Validation Sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)



# Convert to np arrays so that we can use with TensorFlow

X_train = np.array(X_train).astype(np.float32)

X_test  = np.array(X_test).astype(np.float32)

y_train = np.array(y_train).astype(np.float32)

y_test  = np.array(y_test).astype(np.float32)
#Check to make sure split still has 4 features and 3 labels

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
training_size = X_train.shape[1]

test_size = X_test.shape[1]

num_features = 4

num_labels = 3





num_hidden = 10



graph = tf.Graph()

with graph.as_default():

    tf_train_set    = tf.constant(X_train)

    tf_train_labels = tf.constant(y_train)

    tf_valid_set    = tf.constant(X_test)

 

    

    print(tf_train_set)

    print(tf_train_labels)

    

    ## Note, since there is only 1 layer there are actually no hidden layers... but if there were

    ## there would be num_hidden

    weights_1 = tf.Variable(tf.truncated_normal([num_features, num_hidden]))

    weights_2 = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))

    ## tf.zeros Automaticaly adjusts rows to input data batch size

    bias_1 = tf.Variable(tf.zeros([num_hidden]))

    bias_2 = tf.Variable(tf.zeros([num_labels]))

    

    

    logits_1 = tf.matmul(tf_train_set , weights_1 ) + bias_1

    rel_1 = tf.nn.relu(logits_1)

    logits_2 = tf.matmul(rel_1, weights_2) + bias_2

    

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_2, tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(.005).minimize(loss)

    

    

    ## Training prediction

    predict_train = tf.nn.softmax(logits_2)

    

    

    

    # Validation prediction

    logits_1_val = tf.matmul(tf_valid_set, weights_1) + bias_1

    rel_1_val    = tf.nn.relu(logits_1_val)

    logits_2_val = tf.matmul(rel_1_val, weights_2) + bias_2

    predict_valid = tf.nn.softmax(logits_2_val)
def accuracy(predictions, labels):

  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

          / predictions.shape[0])



num_steps = 10000

with tf.Session(graph = graph) as session:

    tf.initialize_all_variables().run()

    print(loss.eval())

    for step in range(num_steps):

        _,l, predictions = session.run([optimizer, loss, predict_train])

        

        if (step % 2000 == 0):

              #print(predictions[3:6])

              print('Loss at step %d: %f' % (step, l))

              print('Training accuracy: %.1f%%' % accuracy( predictions, y_train[:, :]))

              print('Validation accuracy: %.1f%%' % accuracy(predict_valid.eval(), y_test))

        
# Conclusion - Despite poor performance of the model, I was pleased to see the 

# accuaracy for the training and validation set appears to match closely with

# the values quoted for the probabalistic neural network studied in the paper.