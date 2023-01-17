import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt

%matplotlib inline
#load training and testing data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

print("Training data shape: {}".format(train_data.shape))
print("Test data shape: {}".format(test_data.shape))
#for training data separate features and labels
train_data_x = train_data.iloc[:, 1:785]
train_data_y = train_data.iloc[:, 0:1]

print("Shape of training features: {}".format(train_data_x.shape))
print("Shape of training labels: {}".format(train_data_y.shape))
#Normalize features data
train_data_x = train_data_x/255
test_data = test_data/255

#Lets convert our training and testing data from pandas dataframes into numpy 
#arrays needed to train our model

train_x = train_data_x.as_matrix()
train_y = train_data_y.as_matrix()

test_x = test_data.as_matrix()

#Conver output label to one hot vector
train_y = to_categorical(train_y, 10)
print("Shape of training lables: {}".format(train_y.shape))
#Split the training set into training and cross validation set. We will use the cross validation
#set to check how well our model is doing on unseen training example.

train_x, cv_x, train_y, cv_y = train_test_split(train_x, train_y, 
                                                test_size = 2000, random_state = 42)

print("Number of examples in training set: {}".format(train_x.shape[0]))
print("Number of examples in cross validation set: {}".format(cv_x.shape[0]))
#Let's build tensorflow computation graph by adding operations to create model to predict labels, 
#to calculate the accuracy of the model and to calculate loss
tf.reset_default_graph()

#Define placeholders for input and output data
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
#Add Model parameters
w = tf.get_variable("weights", dtype=tf.float32, shape=[784, 10], initializer=tf.random_normal_initializer(stddev=0.01))
b = tf.get_variable("bias", dtype=tf.float32, shape=[1, 10], initializer=tf.zeros_initializer())
#linear model
z = tf.matmul(x, w) + b

y_pred = tf.nn.softmax(logits =z)
#define loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=z)
#Hyperparameters
learning_rate = 0.01
batch_size = 128
epoch = 40
#We will use gradient decent optimizer to minimise the loss
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#Accuracy for the model is defined as the ratio of number of correctly predicted labels 
#to the total number of labels 
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1)), tf.float32))
#Add operation to initialize all varibles 
init_op =  tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    
    #Divide input training set into mini batches of size batch_size
    #If the total number of training examles is not exactly divisible by batch_size, the last batch
    #will have less number of examples than batch_size
    
    total_size = train_x.shape[0]
    number_of_batches = int(total_size/batch_size)
    
    for e in range(epoch):
        epoch_cost = 0
        epoch_accuracy = 0
        for i in range(number_of_batches):
            mini_x = train_x[i*batch_size:(i+1)*batch_size, :]
            mini_y = train_y[i*batch_size:(i+1)*batch_size, :]
            _, cost = sess.run([train_step, loss], feed_dict={x:mini_x, y:mini_y})
            train_accuracy = sess.run(accuracy, feed_dict={x:mini_x, y:mini_y})
            epoch_cost += cost
            epoch_accuracy += train_accuracy
        
        #If the total number of training examles is not exactly divisible by batch_size, 
        #we have one more batch of size (total_size - number_of_batches*batch_size)
        if total_size % batch_size != 0:
            mini_x = train_x[number_of_batches*batch_size:total_size, :]
            mini_y = train_y[number_of_batches*batch_size:total_size, :]
            _, cost = sess.run([train_step, loss], feed_dict={x:mini_x, y:mini_y})
            train_accuracy = sess.run(accuracy, feed_dict={x:mini_x, y:mini_y})
            epoch_cost += cost
            epoch_accuracy += train_accuracy
        
        epoch_cost /= number_of_batches
        
        if total_size % batch_size != 0:
            epoch_accuracy /= (number_of_batches+1)
        else:
            epoch_accuracy /= number_of_batches
        print("Epoch: {} Cost: {} accuracy: {} ".format(e+1, np.squeeze(epoch_cost), epoch_accuracy))
    
    #Cross validation loss and accuracy
    
    cv_loss, cv_accuracy = sess.run([loss, accuracy], {x:cv_x, y:cv_y})
    print("Cross validation loss: {} accuracy: {}".format(np.squeeze(cv_loss), cv_accuracy))
    
    #prediction for test set
    prediction = sess.run(tf.argmax(y_pred, axis=1), {x:test_x})
    
        
#Let us choose random 100 images from test set and see 
#actual label and predicted label by our model for the same
predicted_labels = prediction
permutations = np.random.permutation(28000)

fig, axs = plt.subplots(10, 10, figsize = (20, 20))
for r in range(10):
  for c in range(10):
    axs[r, c].imshow(np.reshape(test_x[permutations[10*r+c]]*255, (28, 28)), cmap='Greys')
    axs[r, c].axis('off')
    axs[r, c].set_title('prediction: '+str(predicted_labels[permutations[10*r+c]]))