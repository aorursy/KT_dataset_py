import tensorflow as tf
import pandas as pd
import numpy as np
# 数据预处理
df = pd.read_csv('../input/train.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:,0].values
from sklearn.preprocessing import OneHotEncoder
y_one_hot = OneHotEncoder().fit_transform(np.expand_dims(y, axis=1)).toarray()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=0)
# 神经网络无需标准化？
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
X_train_std = sd.fit_transform(X_train)
X_test_std = sd.transform(X_test)
X_train = np.reshape(X_train_std, [X_train_std.shape[0], 28, 28, 1])
X_test = np.reshape(X_test_std, [X_test_std.shape[0], 28, 28, 1])
sess = tf.InteractiveSession()
# 输入与输出
X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x-input')
y_ = tf.placeholder(tf.int32, shape=[None, 10], name='y-input')
y_ = tf.cast(y_, 'float')
# 卷积层1
w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], stddev=0.1))
bias_conv1 = tf.Variable(tf.constant(0.1, shape=[6]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(X, w_conv1, padding='SAME', strides=[1,1,1,1]) + bias_conv1)
# 池化层1
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 卷积层2
w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], stddev=0.1))
bias_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, padding='VALID', strides=[1,1,1,1]) + bias_conv2)
# 池化层2
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层1
W_fc1 = tf.Variable(tf.truncated_normal(shape=[5 * 5 * 16, 120], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[120]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层2
W_fc2 = tf.Variable(tf.truncated_normal(shape=[120, 84], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[84]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 输出层
W_fc3 = tf.Variable(tf.truncated_normal(shape=[84, 10], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

# 损失函数与准确率
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
batch_size = 100
data_size = X_train.shape[0]
for i in range(10000):
    start = (i*batch_size) % data_size # 可以重复选取
    end = min(start+batch_size, data_size)
    if i % 1000 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={X: X_train[start:end], y_: y_train[start:end], keep_prob:1.0})
        print('training accuracy after %d step(s) is %.2f' % (i, train_accuracy))
    sess.run(train_step, feed_dict={X: X_train[start:end], y_: y_train[start:end], keep_prob:0.5})
test_accuracy = accuracy.eval(feed_dict={X: X_test, y_: y_test, keep_prob:1.0})
print("test accuracy: %.2f" % test_accuracy)
y_pred = tf.argmax(y_conv,1)
# predict the test.csv and save the result to local file
df_x= pd.read_csv('../input/test.csv')
x = df_x.values
x_std = sd.transform(x)
x_re = np.reshape(x_std, [x_std.shape[0], 28, 28, 1])
y_predict = sess.run(y_pred, feed_dict={X: x_re, keep_prob: 1})
df_result = pd.DataFrame({'ImageId': list(range(1, len(y_predict)+1)), 'Label': y_predict})
df_result.to_csv('../input/result.csv',index=None)
saver = tf.train.Saver()
saver.save(sess, "../input/cnn_model.ckpt")