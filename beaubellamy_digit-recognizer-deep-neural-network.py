# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as mticker

style.use('fivethirtyeight')
# Try image augmentaion (rotation, transformations)
#https://www.kaggle.com/dhayalkarsahilr/easy-image-augmentation-techniques-for-mnist

# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print (type(train.iloc[1,1]))
print (type(test.iloc[1,1]))
label = train["label"]
# Convert to one_hot array
Labels = np.zeros((label.size, label.max()+1))
Labels[np.arange(label.size),label] = 1

# Drop 'label' column
Features = train.drop(labels = ["label"],axis = 1)

trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(Features,Labels,test_size=0.2,random_state=0)
#np.array(trainFeatures)
#np.array(trainLabels)

#trainLabels.head()
print (trainFeatures.shape)
print (trainLabels.shape)
# Set some model parameters
n_nodes = 2500
n_classes = 10

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
keep_prob = tf.placeholder(tf.float32)

# Define the neural network
def neural_network_model(data, keep_prob):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes])),
                      'biases':tf.Variable(tf.random_normal([n_nodes]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                      'biases':tf.Variable(tf.random_normal([n_nodes]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                      'biases':tf.Variable(tf.random_normal([n_nodes]))}

    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                      'biases':tf.Variable(tf.random_normal([n_nodes]))}

    hidden_5_layer = {'weights':tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                      'biases':tf.Variable(tf.random_normal([n_nodes]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l1 = tf.nn.dropout(l1, keep_prob)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l2 = tf.nn.dropout(l2, keep_prob)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    l3 = tf.nn.dropout(l3, keep_prob)

    #l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    #l4 = tf.nn.relu(l4)
    #l4 = tf.nn.dropout(l4, keep_prob)

    #l5 = tf.add(tf.matmul(l4,hidden_5_layer['weights']), hidden_5_layer['biases'])
    #l5 = tf.nn.relu(l5)
    #l5 = tf.nn.dropout(l5, keep_prob)

    # Add dropout layer
    # dropout = tf.matmult(l5,0.6)
    
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    # output = tf.matmul(dropout,output_layer['weights']) + output_layer['biases']
    
    return output


# Run the neural network session.
def train_neural_network(x):
    
    prediction = neural_network_model(x, 0.5)
    cost = tf.reduce_mean( tf.nn.
                softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # adjust learning rate
    # data augmentation
    # 60 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.927381
    
    num_epochs = 50
    # 10 epochs, 1 layers, 500 nodes/layer, Adam Optimizer: 0.12238095
    # 30 epochs, 1 layers, 500 nodes/layer, Adam Optimizer: 0.27964285
    # 50 epochs, 1 layers, 500 nodes/layer, Adam Optimizer: 0.3727381
    
    # 10 epochs, 3 layers, 500 nodes/layer, Adam Optimizer: 0.12464286
    # 30 epochs, 3 layers, 500 nodes/layer, Adam Optimizer: 0.18059523
    # 50 epochs, 3 layers, 500 nodes/layer, Adam Optimizer: 0.2325
    
    # 10 epochs, 3 layers, 1000 nodes/layer, Adam Optimizer: 0.1370238
    # 30 epochs, 3 layers, 1000 nodes/layer, Adam Optimizer: 0.27595237
    # 50 epochs, 3 layers, 1000 nodes/layer, Adam Optimizer: 0.43785715
    
    # 10 epochs, 3 layers, 1500 nodes/layer, Adam Optimizer: 0.1575
    # 30 epochs, 3 layers, 1500 nodes/layer, Adam Optimizer: 0.37238094
    # 50 epochs, 3 layers, 1500 nodes/layer, Adam Optimizer: 0.5221428
    
    # 10 epochs, 5 layers, 500 nodes/layer, Adam Optimizer: 0.10666667
    # 30 epochs, 5 layers, 500 nodes/layer, Adam Optimizer: 0.10607143
    # 50 epochs, 5 layers, 500 nodes/layer, Adam Optimizer: 0.112261906
    
    # 10 epochs, 5 layers, 1000 nodes/layer, Adam Optimizer: 0.103452384
    # 30 epochs, 5 layers, 1000 nodes/layer, Adam Optimizer: 0.12630953
    # 50 epochs, 5 layers, 1000 nodes/layer, Adam Optimizer: 0.13666667
    
    # 10 epochs, 5 layers, 1500 nodes/layer, Adam Optimizer: 0.10916667
    # 30 epochs, 5 layers, 1500 nodes/layer, Adam Optimizer: 0.15214285
    # 50 epochs, 5 layers, 1500 nodes/layer, Adam Optimizer: 0.17392857
    
    # 10 epochs, 3 layers, 2000 nodes/layer, Adam Optimizer: 0.21035714
    # 10 epochs, 5 layers, 2000 nodes/layer, Adam Optimizer: 0.12309524
    
    # 50 epochs, 3 layers, 2000 nodes/layer, Adam Optimizer: 0.60464287
    # 50 epochs, 3 layers, 2500 nodes/layer, Adam Optimizer: 0.64488095
    
    # 30 epochs, 5 layers, 2000 nodes/layer, Adam Optimizer: 0.
    # 30 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 30 epochs, 5 layers, 3000 nodes/layer, Adam Optimizer: 0.
    # 30 epochs, 7 layers, 2000 nodes/layer, Adam Optimizer: 0.
    # 30 epochs, 7 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 30 epochs, 7 layers, 3000 nodes/layer, Adam Optimizer: 0.
    # 30 epochs, 9 layers, 2000 nodes/layer, Adam Optimizer: 0.
    # 30 epochs, 9 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 30 epochs, 9 layers, 3000 nodes/layer, Adam Optimizer: 0.
    
    # 30 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 40 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 50 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 60 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 70 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 80 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 90 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    # 100 epochs, 5 layers, 2500 nodes/layer, Adam Optimizer: 0.
    
    with tf.Session() as sess:
        
        tf.initialize_all_variables().run()
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], 
                    feed_dict={x: trainFeatures, y: trainLabels, keep_prob: 0.5})
            epoch_loss += c

            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:testFeatures, y:testLabels}))
        
        


print (x.shape)
print (trainFeatures.shape)
# Train the neural network
train_neural_network(x)

# Make some predictions
# Convert test set to float
test = test.astype(np.float32, copy=False)
prediction = neural_network_model(test,1.0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    test_pred = sess.run(prediction, feed_dict={x:test, keep_prob: 1})
    test_labels = np.argmax(test_pred, axis=1)

print (test_pred.shape)
print (test_labels.shape)

print (test_pred[0])
print (test_labels[0])
# plot the first 100 test images and see how good we were.
fig = plt.figure(figsize=(15,15))
ax = []
square = 10
for i in range(square*square):
    ax.append(plt.subplot2grid((square,square), (i%square,int(i/square)), rowspan=1, colspan=1))

    pixels = test.iloc[i,:].values
    pixels = pixels.reshape((28, 28))

    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)
    ax[i].xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[i].yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax[i].set_title('{label}'.format(label=test_labels[i]),fontdict={'fontsize':10})
    ax[i].imshow(pixels, cmap='gray')

plt.show()
# Anywhere from 6-10% right. This is crap :(
# Histogram of predicted values
plt.figure(figsize=(12,8))
plt.hist(test_labels, histtype='bar', rwidth=0.8)
plt.xlabel('Predicted Label')
plt.ylabel('Count')
plt.title('Frequency of predictions')
plt.show()


