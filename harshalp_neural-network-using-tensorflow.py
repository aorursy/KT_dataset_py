
#Neural Network Tutorial:
# In this Neural Network tutorial we will take a step forward and will discuss about the network of 
#Perceptrons called Multi-Layer Perceptron (Artificial Neural Network).


#import the required libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split
#function to read the sonar dataset
def read_dataset():
    df = pd.read_csv("../input/sonar.all-data.csv")
    print(len(df.columns))
    X = df[df.columns[1:60]].values
    y=df[df.columns[60]]
    #encode the depedent variable, single it has more than one class
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    return(X,Y,y)
#normalise the features of the dataset
def feature_normalize(features):
    mu = np.mean(features,axis=0)
    sigma = np.std(features,axis=0)
    normalize_features = (features - mu) / sigma
    return(normalize_features)
#appending the bias
def append_bias_reshape(features):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    features = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim+1])
    return features
#define the one hot encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
#plot the graph for the data
def plot_points(features,labels):
    normal = np.where(labels == 0)
    outliers = np.where(labels == 1)
    fig = plt.figure(figsize=(10,8))
    plt.plot(features[normal ,0],features[normal ,1],'bx')
    plt.plot(features[outliers,0],features[outliers ,1],'ro')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()
#read the data
X,Y,y = read_dataset() #X - Features , Y - Labels
normalized_featues = feature_normalize(X)
plot_points(X,y)

#Transform the data in training and testing
X,Y = shuffle(X,Y,random_state=1)
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=42)
#print the shape of the train and test data values
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
#define and initialize the variables to work with the tensors
learning_rate = 0.1
training_epochs = 1000

cost_history = np.empty(shape=[1],dtype=float)
n_dim = X.shape[1]
n_class = 2
x = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
#initialize all variables.
init = tf.global_variables_initializer()
#define the cost function
y_ = tf.placeholder(tf.float32,[None,n_class])
y = tf.nn.softmax(tf.matmul(x, W)+ b)
cost_function = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y)),reduction_indices=[1]))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
#initialize the session
sess = tf.Session()
sess.run(init)
mse_history = []
#calculate the cost for each epoch
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x:train_x,y_:train_y})
    cost = sess.run(cost_function,feed_dict={x: train_x,y_: train_y})
    cost_history = np.append(cost_history,cost)
    pred_y = sess.run(y, feed_dict={x: test_x})
    print('epoch : ', epoch,  ' - ', 'cost: ', cost)

mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ",(sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))

plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()