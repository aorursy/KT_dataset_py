import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
data_train = pd.read_csv('../input/train.csv')
data_train.shape
data_test = pd.read_csv('../input/test.csv')
data_test.shape
x_train = data_train.iloc[:, 1:].values
print("Number of images in training dataset:", x_train.shape[0])
print("Number of pixels in each image in training dataset:", x_train.shape[1])
x_test = data_test.iloc[:, :].values
print("Number of images in test dataset:", x_test.shape[0])
print("Number of pixels in each image in test dataset:", x_test.shape[1])
x_train = x_train.reshape(42000, 28, 28, 1)
x_train.shape
x_test = x_test.reshape(28000, 28, 28, 1)
x_test.shape
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_train.shape, x_test.shape
x_train = x_train.reshape(42000, 32 * 32)
x_test = x_test.reshape(28000, 32 * 32)
x_train.shape, x_test.shape
y_train = data_train.iloc[:, :1].values.flatten()
print('Shape of Training Labels:', y_train.shape)
def one_hot_encode(y, n):
    return np.eye(n)[y]

y_encoded_train = one_hot_encode(y_train, 10)
print('Shape of y_train after encoding:', y_encoded_train.shape)
def next_batch(batch_size, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
def display_images(data, title, display_label = True):
    x, y = data
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    fig.suptitle(title, fontsize = 18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(32, 32), cmap = 'binary')
        if display_label:
            ax.set_xlabel(y[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
def display_filters(x, rows, cols, figsize, title, display_label = True):
    fig, axes = plt.subplots(rows, cols, figsize = figsize)
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    fig.suptitle(title, fontsize = 18)
    c = 1
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(x[i].shape[0], x[i].shape[1]), cmap = 'binary')
        if display_label:
            ax.set_xlabel("Filter " + str(c))
        ax.set_xticks([])
        ax.set_yticks([])
        c += 1
    plt.show()
display_images(next_batch(9, x_train, y_train), 'Training Images')
display_images(next_batch(9, x_test, [None] * len(x_test)), 'Test Images', display_label = False)
z = dict(Counter(list(y_train)))
labels = z.keys()
frequencies = [z[i] for i in labels]
labels = [str(i) for i in z.keys()]

plt.figure(figsize = (14, 7))
plt.bar(labels, frequencies)
plt.title('Frequency Distribution of Alphabets in Training Set', fontsize = 20)
plt.show()
# Training Parameters
learning_rate = 0.001
epochs = 10000
batch_size = 128
display_step = 500
# Network Hyperparameters
n_input = 1024
n_classes = 10
# Placeholders
X = tf.placeholder(tf.float32, shape = [None, n_input]) # Placeholder for Images
Y = tf.placeholder(tf.float32, shape = [None, n_classes]) # Placeholder for Labels
weights = {
    # Convolutional Layer 1: 5x5 filters, 1 input channels, 6 output channels
    'w1' : tf.Variable(tf.random_normal([5, 5, 1, 6])), # Image Size becomes 14x14x6
    # Convolutional Layer 2: 5x5 filters, 6 input channels, 16 output channels
    'w2' : tf.Variable(tf.random_normal([5, 5, 6, 16])), # Image Size becomes 5x5x16
    # Fully Connected Layer 1: 5*5*16 = 400 input channels, 120 output channels
    'w3' : tf.Variable(tf.random_normal([400, 120])),
    # Fully Connected Layer 2: 120 input channels, 84 output channels
    'w4' : tf.Variable(tf.random_normal([120, 84])),
    # Fully Connected Layer 3: 84 input channels, 10 (number of classes) output channels
    'w5' : tf.Variable(tf.random_normal([84, 10]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([6])),
    'b2' : tf.Variable(tf.random_normal([16])),
    'b3' : tf.Variable(tf.random_normal([120])),
    'b4' : tf.Variable(tf.random_normal([84])),
    'b5' : tf.Variable(tf.random_normal([10]))
}
# Wrapper function for creating a Convolutional Layer
def conv2d(x, W, b, strides = 1, padding = 'VALID'):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
# Wrapper function for creating a Pooling Layer
def maxpool2d(x, k=2, padding = 'VALID'):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = padding)
def lenet(x, weight, bias):
    x = tf.reshape(x, shape = [-1, 32, 32, 1])
    
    conv1 = conv2d(x, weight['w1'], bias['b1']) # Convolutional Layer 1
    conv1 = maxpool2d(conv1) # Pooling Layer 1
    
    conv2 = conv2d(conv1, weight['w2'], bias['b2']) # Convolutional Layer 2
    conv2 = maxpool2d(conv2) # Pooling Layer 2
    
    # Fully Connected Layer 1
    # Reshaping output of previous convolutional layer to fit the fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['w3'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weight['w3']), bias['b3']) # Linear Function
    fc1 = tf.nn.relu(fc1) # Activation Function
    
    # Fully Connected Layer 2
    fc2 = tf.add(tf.matmul(fc1, weight['w4']), bias['b4']) # Linear Function
    fc2 = tf.nn.relu(fc2) # Activation Function
    
    out = tf.add(tf.matmul(fc2, weight['w5']), bias['b5']) # Output Layer
    
    return out
logits = lenet(X, weights, biases)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Running Initializer
    sess.run(init)
    cost_hist, acc_hist = [], []
    for epoch in range(1, epochs + 1):
        _x, _y = next_batch(batch_size, x_train, y_encoded_train)
        # Running Optimizer
        sess.run(train_op, feed_dict = { X : _x, Y : _y })
        if epoch % display_step == 0:
            # Calculating Loss and Accuracy on the current Epoch
            loss, acc = sess.run([loss_op, accuracy], feed_dict = { X : _x, Y : _y })
            loss = loss
            cost_hist.append(loss)
            acc_hist.append(acc)
            print('Epoch ' + str(epoch) + ', Cost: ' + str(loss) + ', Accuracy: ' + str(acc * 100) + ' %')
    W = sess.run(weights)
    B = sess.run(biases)
    print('-' * 70)
    print('\nOptimization Finished\n')
    print('Accuracy on Training Data: ' + str(sess.run(accuracy, feed_dict = { X : x_train, Y : y_encoded_train, }) * 100) + ' %')
plt.plot(list(range(len(cost_hist))), cost_hist)
plt.title("Change in cost")
plt.show()
plt.plot(list(range(len(acc_hist))), acc_hist)
plt.title("Change in accuracy")
plt.show()
def convert_layer_to_filters(layer):
    return layer.reshape(layer.shape[-1] * layer.shape[-2], layer.shape[0], layer.shape[1])
filters_1 = convert_layer_to_filters(W['w1'])
filters_2 = convert_layer_to_filters(W['w2'])
filters_1.shape, filters_2.shape
display_filters(filters_1, 1, 6, (14, 3), 'Filters of Layer 1', display_label = True)
display_filters(filters_2, 16, 6, (14, 18), 'Filters of Layer 2', display_label = True)
# KFold cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits = 10)

current_fold = 1
train_acc_hist = []
test_acc_hist = []

for train_index, test_index in kf.split(y_encoded_train):
     
    KFold_X_train = x_train[list(train_index)]
    KFold_X_test = x_train[test_index]
    KFold_Y_train = y_encoded_train[train_index]
    KFold_Y_test = y_encoded_train[test_index]

    # run the graph
    with tf.Session() as sess:
    
        sess.run(init)

        for epoch in range(1, epochs + 1):

            batch_x, batch_y = next_batch(batch_size, KFold_X_train, KFold_Y_train)

            sess.run(train_op, feed_dict = { X : batch_x, Y : batch_y })
    
        train_accuracy = sess.run(accuracy, feed_dict = { X : KFold_X_train, Y :KFold_Y_train }) * 100
        test_accuracy = sess.run(accuracy, feed_dict = { X : KFold_X_test, Y : KFold_Y_test }) * 100
        
        train_acc_hist.append(train_accuracy)
        test_acc_hist.append(test_accuracy)

        print('\nFOLD-' + str(current_fold) + '\n')
        print('Accuracy on train data \t:  {0:.2f} %'.format(train_accuracy))
        print('Accuracy on test data  \t:  {0:.2f} %'.format(test_accuracy))

    current_fold = current_fold +1
    
train_cross_val_score = np.mean(train_acc_hist)    
test_cross_val_score = np.mean(test_acc_hist)


print('\n\nFINAL TRAIN SET K-FOLD CROSS VALIDATION ACCURACY \t:  {0:.2f}'.format(train_cross_val_score))
print('\nFINAL TEST SET K-FOLD CROSS VALIDATION ACCURACY    \t:  {0:.2f}'.format(test_cross_val_score))
for key in W.keys():
    np.save(key, W[key])
for key in B.keys():
    np.save(key, B[key])
# Converting numpy ndarrays into tensorflow variables 
for key in weights.keys():
    weights[key] = tf.Variable(W[key])
for key in biases.keys():
    biases[key] = tf.Variable(B[key])
pred = tf.argmax(lenet(X, weights, biases), 1)
with tf.Session() as sess:
    
    for key in weights.keys():
        sess.run(weights[key].initializer)
    for key in biases.keys():
        sess.run(biases[key].initializer)
    y_pred = sess.run(pred, feed_dict= {X : x_test}) 
display_images(next_batch(9, x_test, y_pred), 'Prediction on Test Images')