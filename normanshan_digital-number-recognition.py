import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import math
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn.model_selection import train_test_split

%matplotlib inline
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# divide training data into images and labels
y_train = train_data['label']
x_train = train_data.drop(columns=['label'])

x_test = test_data
del(train_data)      # free space
x_train.isnull().any().describe()  # dataframe -> series -> description
print(y_train.value_counts())
g = sns.countplot(y_train)
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)
x_train = x_train / 255
x_test = x_test / 255
y_train = to_categorical(y_train, num_classes=10)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size = 0.1,random_state=10)
# 卷积神经网络
# X ->       conv ->   relu -> max pooling -> conv ->   relu -> max pooling -> flatten ->  full connection -> softmax
# 28*28*1    5*5*1*8  28*28*8    4*4          3*3*8*16            2*2         4*4*16        10 neuron

X = tf.placeholder(shape=[None,28,28,1], dtype = tf.float32)
Y = tf.placeholder(shape=[None, 10], dtype = tf.float32)
W1 = tf.get_variable("W1", [5,5,1,8], initializer =  tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", [3,3,8,64], initializer =  tf.contrib.layers.xavier_initializer())
Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
A1 = tf.nn.relu(Z1)
P1 = tf.nn.max_pool(A1, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
A2 = tf.nn.relu(Z2)
P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
flat = tf.contrib.layers.flatten(P2)
Z3 = tf.contrib.layers.fully_connected(flat, 10, activation_fn=None)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z3, labels = Y))
train = tf.train.AdamOptimizer(0.01).minimize(cost)
# Calculate accuracy on the training and validation set
y_predict = tf.nn.softmax(Z3)
y_predict_1D = tf.argmax(tf.nn.softmax(Z3), 1)
y_true_1D = tf.argmax(Y, 1)
correct_prediction = tf.equal(y_predict_1D, y_true_1D)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
num_epoch = 50
batch_size = 128
num_batches = int(x_train.shape[0]/batch_size)
# seperate X&Y into batches
def random_mini_batches(X,Y,mini_batch_size,seed=222):
    mini_batches = []
    #<1> shuffle
    np.random.seed(seed)
    m = x_train.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    #<2> Partition
    num_complete_minibatches = int(math.floor(m/mini_batch_size))  #轮数
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k+1) * mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k+1) * mini_batch_size,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[(k+1) * mini_batch_size:m,:]
        mini_batch_Y = shuffled_Y[(k+1) * mini_batch_size:m,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


init = tf.global_variables_initializer()
cost_train = []
accuracy_train = []
accuracy_val = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range (num_epoch):
        batches = random_mini_batches(x_train,y_train,batch_size)
        epoch_loss = 0
        for batch in batches:
            x_batch,y_batch = batch
            _,loss = sess.run([train,cost], feed_dict={X:x_batch, Y:y_batch})
            epoch_loss += loss / num_batches
        cost_train.append(epoch_loss)
        if (epoch + 1) % 2 == 0:
            print('epoch ' + str(epoch + 1) + ':' + str(epoch_loss/num_batches))
        
        accuracy_train_epoch = sess.run(accuracy,feed_dict={X:x_train, Y:y_train})
        accuracy_val_epoch = sess.run(accuracy,feed_dict={X:x_val, Y:y_val})
        
        accuracy_train.append(accuracy_train_epoch)
        accuracy_val.append(accuracy_val_epoch)
            
    # compute the confusion matrix
    confusion_mtx = sess.run(tf.confusion_matrix(y_true_1D, y_predict_1D, 10),feed_dict={X:x_val,Y:y_val})
    y_true_1D = sess.run(y_true_1D,feed_dict={X:x_val,Y:y_val})
    y_predict_1D = sess.run(y_predict_1D,feed_dict={X:x_val,Y:y_val})
    y_predict = sess.run(y_predict,feed_dict={X:x_val,Y:y_val})
plt.plot(range(1,num_epoch + 1),cost_train)
plt.plot(range(1,num_epoch + 1),accuracy_train,label='accuracy_train')    
plt.plot(range(1,num_epoch + 1),accuracy_val,label='accuracy_val')
plt.legend(ncol=2)
plt.ylim(0.9,1)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Display some error results 
def display_errors(errors_index, img_errors, pred_errors, true_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],true_errors[error]))
            n += 1


## locate error examples
errors = (y_true_1D - y_predict_1D != 0) # Error index [True, false, ..., True]
y_predict_errors = y_predict[errors]
y_predict_1D_errors = y_predict_1D[errors]
y_true_1D_errors = y_true_1D[errors]
x_val_errors = x_val[errors]

## probabilities of the wrong predicted numbers in the error examples
y_predict_errors_predict_prob = np.argmax(y_predict_errors, axis=1)

## probabilities of the true labels in the error examples
y_predict_errors_true_prob = np.diagonal(np.take(y_predict_errors, y_true_1D_errors, axis=1))

## Difference between the probability of the predicted label and the true label
delta_predict_true_errors = y_predict_errors_predict_prob - y_predict_errors_true_prob

## Sorted list of the delta prob errors
sorted_delta_errors = np.argsort(delta_predict_true_errors)

## Top 6 errors 
most_important_errors = sorted_delta_errors[-6:]

## Show the top 6 errors
display_errors(most_important_errors, x_val_errors, y_predict_1D_errors, y_true_1D_errors)
y_predict_1D = pd.Series(y_predict_1D,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_predict_1D],axis = 1)

submission.to_csv("cnn_mnist.csv",index=False)
