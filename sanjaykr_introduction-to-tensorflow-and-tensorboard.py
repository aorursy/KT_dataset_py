import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from tensorflow.python.keras import utils
from tensorflow.python.keras.preprocessing import image
import plotly
import plotly.graph_objs as go
import seaborn as sns
from IPython.display import Image
import matplotlib.cm as cm
import numpy as np
%matplotlib inline
plotly.offline.init_notebook_mode(True)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.isnull().any().any()
test.isnull().any().any()
labels = train['label']
train = train.drop(['label'], axis=1)
f, ax = plt.subplots(figsize=(15,7))
sns.countplot(labels)
def show_img(img):
    img = img.values.reshape(28,28)
    plt.imshow(img,cmap=cm.binary)

show_img(train.iloc[80])
train /= 255.0
test /= 255.0
train = train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
labels = utils.to_categorical(labels)
X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size = 0.1)
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
batch_size = 16
patch_size = 5
image_size = 28
num_labels = 10
num_channels = 1
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False)
  
        # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(X_val, dtype=tf.float32)
    tf_test_dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
  
        # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, 32], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([32]))
    
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, 32, 64], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[64]))
    
    layer3_weights = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[1024]))

    layer4_weights = tf.Variable(tf.truncated_normal([1024, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
    # Model.
    def model(data, dropout):
        conv1 = conv2d(data, layer1_weights, layer1_biases)
        conv1 = maxpool2d(conv1, k=2)
        conv2 = conv2d(conv1, layer2_weights, layer2_biases)
        conv2 = maxpool2d(conv2, k=2)
        fullyconnected1 = tf.reshape(conv2, [-1, layer3_weights.get_shape().as_list()[0]])
        fullyconnected1 = tf.add(tf.matmul(fullyconnected1, layer3_weights), layer3_biases)
        fullyconnected1 = tf.nn.relu(fullyconnected1)
        fullyconnected1 = tf.nn.dropout(fullyconnected1, dropout)
        output = tf.add(tf.matmul(fullyconnected1, layer4_weights), layer4_biases)
        return output
  
    # Training computation.
    logits = model(tf_train_dataset, 0.75)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    learning_rate = tf.train.exponential_decay(0.001, global_step=global_step,
                                           decay_steps = 4000, decay_rate = 0.96, staircase=False)
    
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(loss, global_step= global_step)
    # Predictions for the training, validation, and test data.
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 1))
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
num_steps = 9001
valid_accuracy_list = []
train_accuracy_list = []
step = []
loss_list = []
learning_rate_list = []
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter('./logg',
                                      sess.graph)
    print('Initialized')
    for global_step in range(num_steps):
        offset = (global_step * batch_size) % (Y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :, :, :]
        batch_labels = Y_train[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions,lr = sess.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        loss_list.append(l)
        learning_rate_list.append(lr)
        if (global_step % 1000 ==0):
            valid_accuracy = accuracy(valid_prediction.eval(), Y_val)
            train_accuracy = accuracy(predictions, batch_labels)
            print('Minibatch loss at step %d: %f' % (global_step, l))
            print('Minibatch accuracy: %.1f%%' % train_accuracy)
            print('Validation accuracy: %.1f%%' % valid_accuracy)
            valid_accuracy_list.append(valid_accuracy)
            train_accuracy_list.append(train_accuracy)
            step.append(global_step)
    output = []
    for i in range(28):
        batch_test = test[i*1000:(i+1)*1000 , :]
        output.append(sess.run([test_prediction], feed_dict={tf_test_dataset:batch_test}))
trace1 = go.Scatter(
    y = loss_list
)
layout = go.Layout(
    title='Loss',
    xaxis=dict(
        title='Step',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='loss',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data = [trace1]
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
trace1 = go.Scatter(
    y = learning_rate_list
)
layout = go.Layout(
    title='Learning Rate',
    xaxis=dict(
        title='Step',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Learninig rate',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
data = [trace1]
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig)
trace1 = go.Scatter(
    x = step,
    y = train_accuracy_list,
    name = 'Training Accuracy'
)
trace2 = go.Scatter(
    x = step,
    y = valid_accuracy_list,
    name = 'Validation Accuracy'
)
data = [trace1, trace2]

plotly.offline.iplot(data)
submission = []
for i in range(28):
    for j in range(1000):
        submission.append(np.argmax(output[i][0][j]))
submission = pd.Series(submission, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),submission],axis = 1)

submission.to_csv("submission.csv",index=False)
submission.head()