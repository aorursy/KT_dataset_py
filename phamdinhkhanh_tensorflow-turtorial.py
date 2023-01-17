import tensorflow as tf
a = tf.constant(5)
a
with tf.Session() as sess:
    print(sess.run(a))
with tf.Session() as sess:
    sess.as_default()
    print(a.eval())
tf.InteractiveSession()
a.eval()
# Khởi tạo một giá trị Variable trong tensorflow
v = tf.Variable([1, 2, 3], name = 'vector')
m_2D = tf.Variable([[1, 2], [3, 4]], name = 'matrix_2D')
m_nD = tf.Variable(tf.zeros([2, 2, 2]), name = 'matrix_nD')
# Khởi tạo một giá trị Variable trong tensorflow
gv_v = tf.get_variable(initializer = [1, 2, 3], name = 'vector')
gv_m_2D = tf.get_variable(initializer = [[1, 2], [3, 4]], name = 'matrix_2D')
gv_m_nD = tf.get_variable(initializer = tf.zeros([2, 2, 2]), name = 'matrix_nD')
# Khởi tạo tất cả trong 1 lần:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([gv_v, gv_m_2D, gv_m_nD]))
# Khởi tạo tất cả trong 1 lần:
with tf.Session() as sess:
    sess.run(tf.variables_initializer([v, m_2D, m_nD]))
    print(sess.run([v, m_2D, m_nD]))
x = tf.placeholder(tf.float32, shape = [2, 3])
y = tf.constant([[1], [2], [3]], tf.float32) #lưu ý phải kiểu dữ liệu của y và x phải trùng nhau
y_hat = tf.matmul(x, y)
with tf.Session() as sess:
    print(sess.run([y_hat], feed_dict = {x: [[1, 2, 3], [4, 5, 6]]}))
with tf.Session() as sess:
    print(sess.run([y_hat], feed_dict = {x: [[3, 5, 1], [2, 5, 2]]}))
t_zeros = tf.zeros([2, 3, 2],tf.float32)
with tf.Session() as sess:
    print(sess.run(t_zeros))
t_origin = tf.constant([[[1, 2], 
                         [3, 4]],
                        [[5, 6],
                         [7, 8]]])
t_zeros = tf.zeros_like(t_origin)

with tf.Session() as sess:
    print(sess.run(t_zeros))
t_ones = tf.ones([2, 3, 2],tf.float32)
t_ones_like = tf.ones_like(t_ones)
with tf.Session() as sess:
    print('tensor ones:' + str(sess.run(t_ones)) + '\n')
    print('tensor ones like:' + str(sess.run(t_ones_like)))
t_eye = tf.eye(3, 3, [1], tf.float32)
with tf.Session() as sess:
    print(sess.run(t_eye))
t_random = tf.random_normal([2, 3], mean = 9, stddev = 2)
with tf.Session() as sess:
    print(sess.run(t_random))
t_rand_pois = tf.random_poisson(lam = 2, shape = [2, 3])
t_rand_unif = tf.random_uniform(shape = [2, 3], minval = 0, maxval = 2)

with tf.Session() as sess:
    print(sess.run([t_rand_pois, t_rand_unif]))
x = tf.constant(1)
y = tf.constant(2)
z = tf.add(x, y)
t = tf.add(x, -y)
with tf.Session() as sess:
    print(sess.run([z, t]))
x = tf.constant(1)
y = tf.constant(2)
z = tf.multiply(x, y)
with tf.Session() as sess:
    print(sess.run(z))
import numpy as np

x = tf.constant(np.arange(12), shape = [3, 4], dtype = tf.float32)
y = tf.constant(np.arange(16), shape = [4, 4], dtype = tf.float32)
z = tf.matmul(x, y)

with tf.Session() as sess:
    print(sess.run(z))
import numpy as np

x = tf.constant(np.arange(40), shape = [2, 4, 5], dtype = tf.float32)
y = tf.constant(np.arange(40), shape = [2, 5, 4], dtype = tf.float32) 
z = tf.matmul(x, y)

with tf.Session() as sess:
    print(sess.run(z))
x = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
y = tf.reduce_mean(x)
z = tf.reduce_mean(x, axis = 1, keepdims = True)
with tf.Session() as sess:
    print(sess.run([y, z]))
x = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
y = tf.reduce_min(x)
z = tf.reduce_min(x, axis = 1, keepdims = True)
with tf.Session() as sess:
    print(sess.run([y, z]))
x = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype = tf.float32)
y = tf.exp(x)
with tf.Session() as sess:
    print(sess.run([y]))
x = tf.constant([[[1, -2, 3], [4, 5, -6], [7, -8, 9]]], dtype = tf.float32)
y = tf.nn.relu(x)
with tf.Session() as sess:
    print(sess.run([y]))
x = tf.constant([1, 1, 2, 3, 4, 5, 6, 7, 8], dtype = tf.float32)
y = tf.nn.softmax(x)
with tf.Session() as sess:
    print(sess.run([y]))
x = tf.constant([1, 1, 2, 3, 4, 5, 6, 7, 8], dtype = tf.float32)
y = tf.nn.softmax(x)
with tf.Session() as sess:
    print(sess.run(tf.reduce_sum(y)))
x = tf.constant([1, 9, 0, 2, 3, 4, 5, 6, 7, 8], dtype = tf.float32)
y = tf.argmin(x)
z = tf.argmax(x)
with tf.Session() as sess:
    print(sess.run([y, z]))
x = tf.constant(5)
if x.graph is tf.get_default_graph():
    print('x is one part of default graph')
g = tf.Graph()

with g.as_default():
    x = tf.constant(5)
    if x.graph is g:
        print('x is one part of default graph')
import tensorflow as tf
x = tf.Variable(2, name = 'x_variabel')
y = tf.Variable(4, name = 'y_variabel')
z = tf.multiply(x, y)

#Khởi tạo một writer để lưu graph mặc định vào ổ đĩa
writer = tf.summary.FileWriter('first_graph_logs', tf.get_default_graph())
with tf.Session() as sess:
    #Khởi tạo toàn bộ các biến
    sess.run(tf.global_variables_initializer())
    #Thêm operation z vào graph mặc định
    sess.run(z)
#Đóng writer    
writer.close()
import os
import urllib.request as req
import numpy as np
import tensorflow as tf

#set up parameter
TRAIN = 'iris_training.csv'
TRAIN_URL = 'http://download.tensorflow.org/data/iris_training.csv'
TEST = 'iris_testing.csv'
TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'
input_shape = 4
n_classes = 3

def loadfile(filename, link):
    if not os.path.exists(filename):
        raw = req.urlopen(link).read().decode('utf-8')
        with open(filename, 'w') as f:
            f.write(raw)
            
    data = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=filename,
      target_dtype=np.int,
      features_dtype=np.float32)
    #normalize biến dự báo theo phân phối chuẩn
    mu = np.mean(data.data, axis = 0)
    sigma = (np.std(data.data, axis=0))
    predictor = (data.data - mu) / sigma
    
    #Chuyển biến mục tiêu sang dạng onehot endcoder
#     target = np.eye(len(data.target), n_classes, dtype = np.float32)[data.target]
    target = data.target
    return {'predictor': predictor, 'target': target}

train = loadfile(TRAIN, TRAIN_URL)
test = loadfile(TEST, TEST_URL)


X = tf.placeholder(tf.float32, [None, input_shape])
y = tf.placeholder(tf.int32, [None])
# filename_queue = tf.train.string_input_producer([TRAIN, TEST])
# reader = tf.TextLineReader(skip_header_lines = True)
# key, value = reader.read(filename_queue)

# record_defaults = [[0.], [0.], [0.], [0.], [0.]]

# col1, col2, col3, col4, col5 = tf.decode_csv(
#     value, record_defaults=record_defaults)

# features = tf.stack([col1, col2, col3, col4])


# with tf.Session() as sess:
#   # Start populating the filename queue.
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)

# #   for i in range(120):
#     # Retrieve a single instance:
#   example, label = sess.run([features, col5])

#   coord.request_stop()
#   coord.join(threads)
#https://stackoverflow.com/questions/46264133/weights-and-biases-not-updating-in-tensorflow
weights = {
    'l1': tf.Variable(tf.random_normal([input_shape, 10])),
    'l2': tf.Variable(tf.random_normal([10, 15])),                              
    'l3': tf.Variable(tf.random_normal([15, 20])),
    'out': tf.Variable(tf.random_normal([20, 1]))
}


biases = {
    'l1': tf.Variable(tf.random_normal([1, 10])),
    'l2': tf.Variable(tf.random_normal([1, 15])),                              
    'l3': tf.Variable(tf.random_normal([1, 20])),
    'out': tf.Variable(tf.random_normal([1, 3]))
}
                           
    
def neural_network(X):
    layer1 = tf.nn.relu(tf.add(tf.matmul(X, weights['l1']), biases['l1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['l2']), biases['l2']))
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['l3']), biases['l3']))
    out = tf.nn.softmax(tf.add(tf.matmul(layer3, weights['out']), biases['out']))
    return out

nn = neural_network(X)
learning_rate = 0.5
global_step = tf.Variable(0)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits = nn, labels = y))
grad_op = tf.reduce_sum(tf.gradients(loss_op, nn)[0])
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op, global_step = global_step)
#Tính toán mức độ chính xác của model
match_pred = tf.equal(tf.cast(tf.argmax(nn, 1), tf.int32), y)
acc_op = tf.reduce_mean(tf.cast(match_pred, tf.float32))
import time
#Xây dựng vòng lặp fitting model trên từng batch
batch_size = 100 #Kích thước mỗi batch.
n_steps = 2000 #Số lượng các lượt cập nhật dữ liệu.
print_every = 100 #Khoảng cách lượt cập nhật dữ liệu để in ra kết quả thuật toán.

#Tạo hàm lấy batch tiếp theo. Khi lấy hết đến batch cuối cùng của mẫu sẽ shuffle lại mẫu.
def next_batch(X, batch_size, index = 0):
    start = index
    index += batch_size
    if index > len(X['predictor']):
        perm = np.arange(len(X['predictor']))
        np.random.shuffle(perm)
        X['predictor'] = X['predictor'][perm]
        X['target'] = X['target'][perm]
        start = 0
        index = batch_size
    end = index
    return X['predictor'][start:end], X['target'][start:end], index

with tf.Session() as sess:
    #Khởi tạo toàn bộ các biến.
    sess.run(tf.global_variables_initializer())
    idx = 0
    for step in range(n_steps):
        start_time = time.time()
        batch_x, batch_y, idx = next_batch(train, batch_size = batch_size, index = idx)
        #Thực thi thuật toán gradient descent
        sess.run(train_op, feed_dict = {X: batch_x, y: batch_y})
        loss = sess.run(loss_op, feed_dict = {X:batch_x, y:batch_y})
        acc = sess.run(acc_op, feed_dict = {X:batch_x, y:batch_y})
        grad = sess.run(grad_op, feed_dict = {X:batch_x, y:batch_y})
        duration = time.time() - start_time
        if step % print_every == 0:
            print('Step {}; grads: {};Loss value: {:.8f}; Accuracy: {:.4f}; time: {:.4f} sec'.format(sess.run(global_step), grad, loss, acc, duration))
#             print(sess.run(weights['l1']))
            
    print('Finished training!')
    print('Loss value in test: {:.4f}; Accuracy in test: {:.4f}'.format(sess.run(loss_op, feed_dict = {X:test['predictor'], y:test['target']}),
                                                          sess.run(acc_op, feed_dict = {X:test['predictor'], y:test['target']})))
import os
import urllib.request as req
import numpy as np
import tensorflow as tf

#set up parameter
TRAIN = 'iris_training.csv'
TRAIN_URL = 'http://download.tensorflow.org/data/iris_training.csv'
TEST = 'iris_testing.csv'
TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'
input_shape = 4
n_classes = 3

def loadfile(filename, link):
    if not os.path.exists(filename):
        raw = req.urlopen(link).read().decode('utf-8')
        with open(filename, 'w') as f:
            f.write(raw)
    data = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=filename,
      target_dtype=np.int,
      features_dtype=np.float32)
    #normalize biến dự báo theo phân phối chuẩn
    mu = np.mean(data.data, axis = 0)
    sigma = (np.std(data.data, axis=0))
    predictor = (data.data - mu) / sigma
    
    #Chuyển biến mục tiêu sang dạng onehot endcoder
#     target = np.eye(len(data.target), n_classes, dtype = np.float32)[data.target]
    target = data.target
    return {'predictor': predictor, 'target': target}

train = loadfile(TRAIN, TRAIN_URL)
test = loadfile(TEST, TEST_URL)


X = tf.placeholder(tf.float32, [None, input_shape])
y = tf.placeholder(tf.int32, [None])
#https://stackoverflow.com/questions/46264133/weights-and-biases-not-updating-in-tensorflow
weights = {
    'l1': tf.Variable(tf.random_normal([input_shape, 10])),
    'l2': tf.Variable(tf.random_normal([10, 15])),                              
    'l3': tf.Variable(tf.random_normal([15, 20])),
    'out': tf.Variable(tf.random_normal([20, 1]))
}


biases = {
    'l1': tf.Variable(tf.random_normal([1, 10])),
    'l2': tf.Variable(tf.random_normal([1, 15])),                              
    'l3': tf.Variable(tf.random_normal([1, 20])),
    'out': tf.Variable(tf.random_normal([1, 3]))
}
                           
    
def neural_network(X):
    layer1 = tf.nn.relu(tf.add(tf.matmul(X, weights['l1']), biases['l1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['l2']), biases['l2']))
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['l3']), biases['l3']))
    out = tf.nn.softmax(tf.nn.relu(tf.add(tf.matmul(layer3, weights['out']), biases['out'])))
    return out

nn = neural_network(X)



learning_rate = 0.5
global_step = tf.Variable(0)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits = nn, labels = y))
grad_op = tf.reduce_sum(tf.gradients(loss_op, nn)[0])
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op, global_step = global_step)



match_pred = tf.equal(tf.cast(tf.argmax(nn, 1), tf.int32), y)
acc_op = tf.reduce_mean(tf.cast(match_pred, tf.float32))



import time
#Xây dựng vòng lặp fitting model trên từng batch
batch_size = 120 #Kích thước mỗi batch.
n_steps = 2000 #Số lượng các lượt cập nhật dữ liệu.
print_every = 100 #Khoảng cách lượt cập nhật dữ liệu để in ra kết quả thuật toán.

#Tạo hàm lấy batch tiếp theo. Khi lấy hết đến batch cuối cùng của mẫu sẽ shuffle lại mẫu.
def next_batch(X, batch_size, index = 0):
    start = index
    index += batch_size
    if index > len(X['predictor']):
        perm = np.arange(len(X['predictor']))
        np.random.shuffle(perm)
        X['predictor'] = X['predictor'][perm]
        X['target'] = X['target'][perm]
        start = 0
        index = batch_size
    end = index
    return X['predictor'][start:end], X['target'][start:end], index

with tf.Session() as sess:
    #Khởi tạo toàn bộ các biến.
    sess.run(tf.global_variables_initializer())
    idx = 0
    for step in range(n_steps):
        start_time = time.time()
        batch_x, batch_y, idx = next_batch(train, batch_size = batch_size, index = idx)
        #Thực thi thuật toán gradient descent
        sess.run(train_op, feed_dict = {X: batch_x, y: batch_y})
        loss = sess.run(loss_op, feed_dict = {X:batch_x, y:batch_y})
        acc = sess.run(acc_op, feed_dict = {X:batch_x, y:batch_y})
        grad = sess.run(grad_op, feed_dict = {X:batch_x, y:batch_y})
        duration = time.time() - start_time
        if step % print_every == 0:
            print('Step {}; grads: {};Loss value: {:.8f}; Accuracy: {:.4f}; time: {:.4f} sec'.format(sess.run(global_step), grad, loss, acc, duration))
#             print(sess.run(weights['l1']))
            
    print('Finished training!')
    print('Loss value in test: {:.4f}; Accuracy in test: {:.4f}'.format(sess.run(loss_op, feed_dict = {X:test['predictor'], y:test['target']}),
                                                          sess.run(acc_op, feed_dict = {X:test['predictor'], y:test['target']})))