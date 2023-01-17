# Importing libraries
import tensorflow as tf
import numpy as np
import pandas as pd
# Setting seed to remove randomness
np.random.seed(123)
# Importing data
traind = pd.read_csv("../input/train.csv")
testd = pd.read_csv("../input/test.csv")

# Printing data
print("\nTrain data shape: ", traind.shape)
print("\nTest data shape: ", testd.shape)
# Looking at top 5 lines from the train data
traind.head()
# Looking at top 5 lines from the test data
testd.head()
# Importing libraries for plotting images
import matplotlib.pyplot as plt
import math

# Plotting images
fig = plt.figure()

for i in range(1,26):
    ax = fig.add_subplot(5,5,i)
    newdata = traind.iloc[i, 1:len(traind)].values
    grid = newdata.reshape(int(math.sqrt(len(traind.iloc[1, 1:len(traind)]))),int(math.sqrt(len(traind.iloc[1, 1:len(traind)]))))
    ax.imshow(grid)

plt.savefig("Digits.png")
# Converting data frame into numpy arrays
traindata = traind.iloc[:,1:]
traindata = traindata.values
testd = testd.values
# One hot encoding for train labels and then converting them into numpy arrays
import sklearn.preprocessing
lb = sklearn.preprocessing.LabelBinarizer()
lb.fit(range(max(traind.iloc[:,0])+1))
train_label = lb.transform(traind.iloc[:,0])
# Converting all sets into float32 as this is default for tensorflow
traindata = traindata.astype(np.float32)
testd = testd.astype(np.float32)
train_label = train_label.astype(np.float32)
# Defined the no. of nodes, labels, batch_size
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = traind["label"].max()+1
batch_size = 100

# Defining input and output using placeholder in tensorflow
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# Creating architecture for the NN
def neural_network(data):
    
    hidden_layer1 = {"weights":tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                     "bias":tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_layer2 = {"weights":tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    "bias":tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_layer3 = {"weights":tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                     "bias":tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                     "bias":tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, hidden_layer1["weights"]), hidden_layer1["bias"])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_layer2["weights"]), hidden_layer2["bias"])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_layer3["weights"]), hidden_layer3["bias"])
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l3, output_layer["weights"]), output_layer["bias"])
    
    return output
# Training NN
def train_nn(x):
    prediction = neural_network(x)
    
    # Cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

    # Optimizer 
    optimizer = tf.train.RMSPropOptimizer(learning_rate=.001).minimize(cost)

    # Feed forward + BackProp
    epochs = 10
    
    # Running Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        # Steps required for batches
        batch = int(traindata.shape[0]/batch_size)
        
        # Feed forward + BackProp        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # For each epoch we are dividing data into batches and minimizing cost function using optimizer
            for i in range(0, len(traindata), batch):
                epoch_x = traindata[i: i + batch]
                epoch_y = train_label[i: i + batch]
                i, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", epochs, "with loss of", epoch_loss)
            
        # Predicitng accuracy of train model
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                
        print("Accuracy of train model: ", accuracy.eval({x: traindata, y: train_label}))
        
        # Predicting train model on test data
        test_pred = tf.argmax(prediction, 1)
        test_pred = sess.run(test_pred, feed_dict = {x: testd})
        
        # Creating dataframe to store the test result
        result = pd.DataFrame({'ImageId': range(1,len(test_pred)+1), 'Label': test_pred})

    return result
# To run neural network
result = train_nn(x)
# Output
result.to_csv("NN_with_TF.csv", index = False)