import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np 

import tensorflow as tf

import sklearn as skl

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
###### 1) Visualizing the Data with t-SNE

# Load the dataset

tsne_data = pd.read_csv("../input/UCI_Credit_Card.csv")

tsne_data.rename(columns = {'default.payment.next.month':'default'}, inplace=True)





#Set df4 equal to a set of a sample of 1000 deafault and 1000 non-default observations.

df2 = tsne_data[tsne_data.default == 0].sample(n = 1000)

df3 = tsne_data[tsne_data.default == 1].sample(n = 1000)

df4 = pd.concat([df2, df3], axis = 0)



#Scale features to improve the training ability of TSNE.

standard_scaler = StandardScaler()

df4_std = standard_scaler.fit_transform(df4)



#Set y equal to the target values.

y = df4.ix[:,-1].values



tsne = TSNE(n_components=2, random_state=0)

x_test_2d = tsne.fit_transform(df4_std)



#Build the scatter plot with the two types of transactions.

color_map = {0:'red', 1:'blue'}

plt.figure()

for idx, cl in enumerate(np.unique(y)):

    plt.scatter(x = x_test_2d[y==cl,0], 

                y = x_test_2d[y==cl,1], 

                c = color_map[idx], 

                label = cl)

plt.xlabel('X in t-SNE')

plt.ylabel('Y in t-SNE')

plt.legend(loc='upper left')

plt.title('t-SNE visualization of test data')

plt.show()
###### 2) Exploring the Data

# Load the dataset

df = pd.read_csv("../input/UCI_Credit_Card.csv")

df.rename(columns = {'default.payment.next.month':'default'}, inplace=True)

df.head()
df.isnull().sum()
print ("Default :")

print (df.AGE[df.default == 1].describe())

print ()

print ("NO default :")

print (df.AGE[df.default == 0].describe())
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 20



ax1.hist(df.AGE[df.default == 1], bins = bins)

ax1.set_title('Default')



ax2.hist(df.AGE[df.default == 0], bins = bins)

ax2.set_title('No Default')



plt.xlabel('Age')

plt.ylabel('Number of Observations')

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 6 # as many as the education types for simplicity 



ax1.hist(df.EDUCATION[df.default == 1], bins = bins)

ax1.set_title('Default')



ax2.hist(df.EDUCATION[df.default == 0], bins = bins)

ax2.set_title('No Default')



plt.xlabel('Education')

plt.ylabel('Number of Observations')

plt.show()
#Create a new Class for Non Default observations.

df.loc[df.default == 0, 'nonDefault'] = 1

df.loc[df.default == 1, 'nonDefault'] = 0



print(df.default.value_counts())

print()

print(df.nonDefault.value_counts())
#Create dataframes of only default and nonDefault observations.

Default = df[df.default == 1]

NonDefault = df[df.nonDefault == 1]



# Set X_train equal to 80% of the observations that defaulted.

X_train = Default.sample(frac=0.8)

count_Defaults = len(X_train)



# Add 80% of the not-defaulted observations to X_train.

X_train = pd.concat([X_train, NonDefault.sample(frac = 0.8)], axis = 0)



# X_test contains all the observations not in X_train.

X_test = df.loc[~df.index.isin(X_train.index)]



#Shuffle the dataframes so that the training is done in a random order.

X_train = shuffle(X_train)

X_test = shuffle(X_test)



#Add our target classes to y_train and y_test.

y_train = X_train.default

y_train = pd.concat([y_train, X_train.nonDefault], axis=1)



y_test = X_test.default

y_test = pd.concat([y_test, X_test.nonDefault], axis=1)



#Drop target classes from X_train and X_test.

X_train = X_train.drop(['default','nonDefault'], axis = 1)

X_test = X_test.drop(['default','nonDefault'], axis = 1)



#Check to ensure all of the training/testing dataframes are of the correct length

print(len(X_train))

print(len(y_train))

print(len(X_test))

print(len(y_test))



# CHECKED !



#Names of all of the features in X_train.

features = X_train.columns.values



#Transform each feature in features so that it has a mean of 0 and standard deviation of 1; 

#this helps with training the neural network.

for feature in features:

    mean, std = df[feature].mean(), df[feature].std()

    X_train.loc[:, feature] = (X_train[feature] - mean) / std

    X_test.loc[:, feature] = (X_test[feature] - mean) / std

    

# Split the testing data into validation and testing sets

split = int(len(y_test)/2)



inputX = X_train.as_matrix()

inputY = y_train.as_matrix()

inputX_valid = X_test.as_matrix()[:split]

inputY_valid = y_test.as_matrix()[:split]

inputX_test = X_test.as_matrix()[split:]

inputY_test = y_test.as_matrix()[split:]
# Number of input nodes.

input_nodes = 24



# Multiplier maintains a fixed ratio of nodes between each layer.

mulitplier = 3 



# Number of nodes in each hidden layer

hidden_nodes1 = 24

hidden_nodes2 = round(hidden_nodes1 * mulitplier)

hidden_nodes3 = round(hidden_nodes2 * mulitplier)



# Percent of nodes to keep during dropout.

pkeep = tf.placeholder(tf.float32)



# input

x = tf.placeholder(tf.float32, [None, input_nodes])



# layer 1

W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev = 0.15))

b1 = tf.Variable(tf.zeros([hidden_nodes1]))

y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)



# layer 2

W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev = 0.15))

b2 = tf.Variable(tf.zeros([hidden_nodes2]))

y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)



# layer 3

W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev = 0.15)) 

b3 = tf.Variable(tf.zeros([hidden_nodes3]))

y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)

y3 = tf.nn.dropout(y3, pkeep)



# layer 4

W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev = 0.15)) 

b4 = tf.Variable(tf.zeros([2]))

y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)



# output

y = y4

y_ = tf.placeholder(tf.float32, [None, 2])



# Parameters

training_epochs = 20 # These proved to be enough to let the network learn

training_dropout = 0.9

display_step = 1 # 10 

n_samples = y_train.shape[0]

batch_size = 2048

learning_rate = 0.01



# Cost function: Cross Entropy

cost = -tf.reduce_sum(y_ * tf.log(y))



# We will optimize our model via AdamOptimizer

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



# Correct prediction if the most likely value (default or non Default) from softmax equals the target value.

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



###### Train the network

accuracy_summary = [] # Record accuracy values for plot

cost_summary = [] # Record cost values for plot

valid_accuracy_summary = [] 

valid_cost_summary = [] 

stop_early = 0 # To keep track of the number of epochs before early stopping



# Initialize variables and tensorflow session

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    

    for epoch in range(training_epochs): 

        for batch in range(int(n_samples/batch_size)):

            batch_x = inputX[batch*batch_size : (1+batch)*batch_size]

            batch_y = inputY[batch*batch_size : (1+batch)*batch_size]



            sess.run([optimizer], feed_dict={x: batch_x, 

                                             y_: batch_y,

                                             pkeep: training_dropout})



        # Display logs after every 10 epochs

        if (epoch) % display_step == 0:

            train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX, 

                                                                            y_: inputY,

                                                                            pkeep: training_dropout})



            valid_accuracy, valid_newCost = sess.run([accuracy, cost], feed_dict={x: inputX_valid, 

                                                                                  y_: inputY_valid,

                                                                                  pkeep: 1})



            print ("Epoch:", epoch,

                   "Acc =", "{:.5f}".format(train_accuracy), 

                   "Cost =", "{:.5f}".format(newCost),

                   "Valid_Acc =", "{:.5f}".format(valid_accuracy), 

                   "Valid_Cost = ", "{:.5f}".format(valid_newCost))

            

            # Record the results of the model

            accuracy_summary.append(train_accuracy)

            cost_summary.append(newCost)

            valid_accuracy_summary.append(valid_accuracy)

            valid_cost_summary.append(valid_newCost)

            

            # If the model does not improve after 15 logs, stop the training.

            if valid_accuracy < max(valid_accuracy_summary) and epoch > 100:

                stop_early += 1

                if stop_early == 15:

                    break

            else:

                stop_early = 0

            

    print()

    print("Optimization Finished!")

    print()   
# Plot the accuracy and cost summaries 

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))



ax1.plot(accuracy_summary) # blue

ax1.plot(valid_accuracy_summary) # green

ax1.set_title('Accuracy')



ax2.plot(cost_summary)

ax2.plot(valid_cost_summary)

ax2.set_title('Cost')



plt.xlabel('Epochs (x10)')

plt.show()