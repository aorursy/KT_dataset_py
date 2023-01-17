import numpy as np
import pandas as pd
import os
import tensorflow as tf

INPUT_COLUMNS = ["body_acc_x_","body_acc_y_","body_acc_z_","body_gyro_x_","body_gyro_y_","body_gyro_z_","total_acc_x_","total_acc_y_",
                 "total_acc_z_"]
LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

print(os.listdir('../input/uci-har-dataset/uci har dataset/UCI HAR Dataset/train/'))
DATA_DIR='../input/uci-har-dataset/uci har dataset/UCI HAR Dataset/'
TRAIN='train/Inertial Signals/'
TEST='test/Inertial Signals/'
X_TRAIN_PATHS=[DATA_DIR+TRAIN+col+'train.txt' for col in INPUT_COLUMNS]
X_TEST_PATHS=[DATA_DIR+TEST+col+'test.txt' for col in INPUT_COLUMNS]

X_Train = []

for path in X_TRAIN_PATHS:
    file = open(path, 'r')
    X_Train.append([np.array(s, dtype=np.float32) for s in [row.replace('  ', ' ').strip().split(' ') for row in file]])
    file.close()

X_Train=np.transpose(np.array(X_Train), (1, 2, 0))

X_Test = []

for path in X_TEST_PATHS:
    file = open(path, 'r')
    X_Test.append([np.array(s, dtype=np.float32) for s in [row.replace('  ', ' ').strip().split(' ') for row in file]])
    file.close()

X_Test=np.transpose(np.array(X_Test), (1, 2, 0))
Y_TRAIN_PATH=DATA_DIR+'train/y_train.txt'
Y_TEST_PATH=DATA_DIR+'test/y_test.txt'

y_Train=[]
file = open(Y_TRAIN_PATH, 'r')
y_ = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)
file.close()
y_Train= y_ - 1

y_Test=[]
file = open(Y_TEST_PATH, 'r')
y_ = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)
file.close()
y_Test= y_ - 1
TRAIN_LEN = len(X_Train)  
TEST_LEN = len(X_Test)  
NUM_STEPS = len(X_Train[0]) 
NUM_INPUT = len(X_Train[0][0]) 
NUM_HIDDEN = 32
NUM_CLASSES = 6
LR = 0.0025
LAMBDA = 0.0020
NUM_ITERS = TRAIN_LEN * 100  
BATCH_SIZE = 1024
DISP_ITER = 20000  

print(X_Test.shape, y_Test.shape, X_Train.shape, X_Train.shape)
def LSTM_RNN(_X, _weights, _biases):
    _X = tf.transpose(_X, [1, 0, 2])
    _X = tf.reshape(_X, [-1, NUM_INPUT]) 
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    _X = tf.split(_X, NUM_STEPS, 0) 
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    lstm_last_output = outputs[-1]
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def next_batch(_train, step, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]
x = tf.placeholder(tf.float32, [None, NUM_STEPS, NUM_INPUT])
y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
weights = {'hidden': tf.Variable(tf.random_normal([NUM_INPUT, NUM_HIDDEN])),'out': tf.Variable(tf.random_normal([NUM_HIDDEN, NUM_CLASSES], mean=1.0))}
biases = {'hidden': tf.Variable(tf.random_normal([NUM_HIDDEN])),'out': tf.Variable(tf.random_normal([NUM_CLASSES]))}

pred = LSTM_RNN(x, weights, biases)

l2 = LAMBDA * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []


sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

step = 1
while step * BATCH_SIZE <= NUM_ITERS:
    batch_xs = next_batch(X_Train, step, BATCH_SIZE)
    batch_ys = one_hot(next_batch(y_Train, step, BATCH_SIZE))

    
    _, loss, acc = sess.run([optimizer, cost, accuracy],feed_dict={x: batch_xs, y: batch_ys})
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    
    if (step*BATCH_SIZE % DISP_ITER == 0) or (step == 1) or (step * BATCH_SIZE > NUM_ITERS):
        print("Iteration " + str(step*BATCH_SIZE) +  ":Batch Loss = " + "{:.3f}".format(loss) + ", Accuracy = {}".format(acc))
        loss, acc = sess.run([cost, accuracy], feed_dict={x: X_Test,y: one_hot(y_Test)})
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("Test: " + "Batch Loss = {}".format(loss) + ", Accuracy = {}".format(acc))
    step += 1





one_hot_predictions, accuracy, final_loss = sess.run([pred, accuracy, cost],feed_dict={x: X_Test,y: one_hot(y_Test)})

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("RESULT: " + "Batch Loss = {}".format(final_loss) + ", Accuracy = {}".format(accuracy))


saver.save(sess, "./model.ckpt")
print("Model saved")

