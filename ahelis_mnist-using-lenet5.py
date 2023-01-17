import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.contrib.layers import flatten
import os
print(os.listdir("../input"))
train_X = pd.read_csv("../input/train.csv")
train_X.head()
train_Y = train_X['label'].values
train_X = train_X.drop('label',axis=1).values
train_X.shape
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(train_X, train_Y, test_size = 0.3)
train_x = train_x/255.0  #normalization of training set
test_x = test_x/255.0  #normalization of test set

train_x = train_x.reshape(-1,28,28,1)
test_x = test_x.reshape(-1,28,28,1)

train_y = np.eye(10)[train_y]  #one hot encoding
test_y = np.eye(10)[test_y]
#pad image with zeros
train_x      = np.pad(train_x, ((0,0),(2,2),(2,2),(0,0)), 'constant')
test_x       = np.pad(test_x, ((0,0),(2,2),(2,2),(0,0)), 'constant')
print('Shape of train_x \t: ', train_x.shape)
print('Shape of train_y \t: ', train_y.shape)
print('Shape of test_x  \t: ', test_x.shape)
print('Shape of test_y  \t: ', test_y.shape)
def batch(batch_size, x, y):
    index = np.arange(len(x))
    np.random.shuffle(index)
    
    b_index = index[0:batch_size]
    
    x_next = [x[i] for i in b_index]
    y_next = [y[i] for i in b_index]
    
    x_next = np.asarray(x_next)
    y_next = np.asarray(y_next)
    return x_next,y_next
def conv_layer(x, w, b, stride=1, padding='VALID'):
    x = tf.nn.conv2d(x,w,strides=[1,stride,stride,1], padding = padding)
    x = tf.nn.bias_add(x,b)
    x = tf.nn.tanh(x)
    return x
def pool_layer(x, f=2, stride=2, padding = 'VALID'):
    x = tf.nn.avg_pool(x, ksize=[1,f,f,1], strides=[1,stride,stride,1], padding=padding)
    x = tf.nn.tanh(x)
    return x
def fc_layer(x,w,b):
    x = tf.matmul(x,w)
    x = tf.nn.bias_add(x,b)
    return x
# hyperparameters
learning_rate = 0.001
batch_size = 128
epoch = 10000
# image dimensions and number of classes
image_width = 32
image_height = 32
color_channels = 1
n_classes = 10
# placeholders
X = tf.placeholder(tf.float32, shape = [None, image_width, image_height, color_channels]) 
Y = tf.placeholder(tf.float32, shape = [None, n_classes]) 
# parameters
weights =  { 'w1' : tf.Variable(tf.truncated_normal([5,5,1,6], stddev=0.1)),
             'w2' : tf.Variable(tf.truncated_normal([5,5,6,16], stddev=0.1)),
             'w3' : tf.Variable(tf.truncated_normal([400,120], stddev=0.1)),
             'w4' : tf.Variable(tf.truncated_normal([120,84], stddev=0.1)),
             'w5' : tf.Variable(tf.truncated_normal([84,10], stddev=0.1))}
            
biases = {'b1' : tf.Variable(tf.truncated_normal([6], stddev=0.1)),
          'b2' : tf.Variable(tf.truncated_normal([16], stddev=0.1)),
          'b3' : tf.Variable(tf.truncated_normal([120], stddev=0.1)),
          'b4' : tf.Variable(tf.truncated_normal([84], stddev=0.1)),
          'b5' : tf.Variable(tf.truncated_normal([10], stddev=0.1))}         
def Network(x,weights,biases):
    conv1 = conv_layer(x,weights['w1'],biases['b1']) # input - 32x32x1, output - 28x28x6
    pool1 = pool_layer(conv1) # input - 28x28x16, output - 14x14x6
    conv2 = conv_layer(pool1,weights['w2'],biases['b2']) # input - 14x14x6, output - 10x10x16
    pool2 = pool_layer(conv2) # input - 10x10x16, output - 5x5x16
    
    flat = flatten(pool2) # input - 5x5x16, output - 400
    
    fc1 = fc_layer(flat,weights['w3'],biases['b3']) # input - 400, output - 120
    fc1 = tf.nn.tanh(fc1)
    fc2 = fc_layer(fc1,weights['w4'],biases['b4']) # input - 120, output - 84
    fc2 = tf.nn.tanh(fc2)
    fc3 = fc_layer(fc2,weights['w5'],biases['b5']) # input - 84, output - 10
    
    return fc3
logits = Network(X,weights,biases) #model output

# softmax with cross entropy loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) # optimizer
train = optimizer.minimize(loss) # minimize loss
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    
    sess.run(init)
    
    cost_hist, acc_hist = [], []
    
    for epoch in range(1, epoch + 1):
        
        batch_x, batch_y = batch(batch_size, train_x, train_y)
        
        sess.run(train, feed_dict = { X : batch_x, Y : batch_y})
    
        if epoch % 500 == 0:
            c, acc = sess.run([loss, accuracy], feed_dict = { X : batch_x, Y : batch_y})
            cost_hist.append(c)
            acc_hist.append(acc)
            print('Epoch ' + str(epoch) + ', Cost: ' + str(c) + ', Accuracy: ' + str(acc))

    W = sess.run(weights)
    B = sess.run(biases)
    print('-' * 70)
    print('\nOptimization Finished\n')
    print('Accuracy on train data \t: ' + str(sess.run(accuracy, feed_dict = { X : train_x, Y :train_y}) * 100) + ' %')
    print('Accuracy on test data  \t: ' + str(sess.run(accuracy, feed_dict = { X : test_x, Y : test_y}) * 100) + ' %')
eps = list(range(500, epoch+500, 500))
plt.plot(eps, cost_hist)
plt.title("Change in cost")
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()
eps = list(range(500, epoch+500, 500))
plt.plot(eps, acc_hist)
plt.title("Change in accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
test = pd.read_csv("../input/test.csv")

#set parameters
for key in weights.keys():
    weights[key] = tf.Variable(W[key])
for key in biases.keys():
    biases[key] = tf.Variable(B[key])
    
x = test.values.reshape(-1,28,28,1) # reshape
x = x/255.0 # normalize
x = np.pad(x, ((0,0),(2,2),(2,2),(0,0)), 'constant') # pad with zeros, output - 32x32x1 images
logits = Network(X, weights, biases)
with tf.Session() as sess:
    
    for key in weights.keys():
        sess.run(weights[key].initializer)
    for key in biases.keys():
        sess.run(biases[key].initializer)
    
    output = sess.run(logits, feed_dict= {X : x}) 
predictions = np.argmax(output, axis=1)
index = test.index +1 

ImageId = list(index)
Label = list(predictions)

df = pd.DataFrame([ImageId ,Label])
df = df.T
df.columns = [['ImageId', 'Label']]

df.to_csv('submission.csv', index=False)
