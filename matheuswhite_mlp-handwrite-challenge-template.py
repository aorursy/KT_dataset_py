import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, accuracy_score
def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
    
    # normalize x
    X_train = X_train.astype(float) / 255.0
    X_test = X_test.astype(float) / 255.0

    # we reserve the last 10,000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset()
print(X_train.shape, y_train.shape)
plt.imshow(X_train[0], cmap="Greys");

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1]*X_val.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
print(f'Training dim: {X_train.shape}, Val dim: {X_val.shape}, Test dim: {X_test.shape}')

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_val = lb.transform(y_val)
y_test = lb.transform(y_test)
print(f'Train lbl dim: {y_train.shape}, Val lbl dim: {y_val.shape}, Test lbl dim: {y_test.shape}')
num_classes = y_train.shape[1]
num_features = X_train.shape[1]
num_output = y_train.shape[1]
num_layer_0 = 512
num_layer_1 = 256
starter_learning_rate = 1e-3
regularizer_rate = 1e-1  
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework.ops import Tensor

# *****Tips*****
# 1. To achieve high accuracy use stddev=1/sqrt(float(num_layer_inputs)), for variable tensors
# 2. Note that the W and b are tuple of tensors
def mlp(num_inputs: int, num_outputs: int, num_layer_1: int, num_layer_2: int) -> (Tensor, Tensor, Tensor, tuple, tuple):
    X = ___
    y = ___
    y_pred = ___
    W = ___
    b = ___

    return X, y, y_pred, W, b
X, y, y_pred, w, b = mlp(num_features, num_output, num_layer_0, num_layer_1)
bias_sum = tf.reduce_sum(tf.square(b[0]))
for i in range(1, len(b)-1):
    bias_sum = bias_sum + tf.reduce_sum(tf.square(b[i]))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y)) \
      + regularizer_rate * bias_sum
var_list = w + b
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)

correct_prediction = tf.equal(tf.argmax(y_train, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
import csv

def save_y_pred(filename, y):
    with open(filename + '.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Id', 'Category'])
        for index in range(y_test.shape[0]):
            csv_writer.writerow([str(index), str(y[index])])
batch_size = 128
# *****Tips*****
# 1. For debug, use epochs=1. After model is done, use epochs=12
epochs = 12

training_accuracy = []
training_loss = []
testing_accuracy = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        arr = np.arange(X_train.shape[0])
        np.random.shuffle(arr)
        for i in range(0, X_train.shape[0], batch_size):
            sess.run(optimizer, {X: X_train[arr[i:i+batch_size]], y: y_train[arr[i:i+batch_size]]})
        training_accuracy.append(sess.run(accuracy, feed_dict={X: X_train, y: y_train}))
        training_loss.append(sess.run(loss, {X: X_train, y: y_train}))

        y_pred_val = sess.run(y_pred, {X: X_test})

        save_y_pred('y_pred', y_pred_val.argmax(1))

        testing_accuracy.append(accuracy_score(y_test.argmax(1), y_pred_val.argmax(1)))
        print(f'Ephoch: {e}, Train loss: {training_loss[e]:.2f}, Train acc: {training_accuracy[e]:.3f}, Test acc: {testing_accuracy[e]:.3f}')