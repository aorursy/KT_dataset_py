from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
# w表示每一个特征值（像素点）会影响结果的权重
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
# 是图片实际对应的值
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# mnist.train 训练数据
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))

prediction = tf.argmax(y,1) 
label = tf.argmax(y_,1) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                   y_: mnist.test.labels}))
test_index = 10
test_image = mnist.test.images[test_index]
test_label = mnist.test.labels[test_index]
print('预测的值是：',sess.run(prediction, feed_dict={x: np.array([test_image]), y_: np.array([test_label])}))
print('实际的值是：',sess.run(label,feed_dict={x: np.array([test_image]), y_: np.array([test_label])}))
one_pic_arr = np.reshape(test_image, (28, 28))
pic_matrix = np.matrix(one_pic_arr, dtype="float")
plt.imshow(pic_matrix)
plt.show()
for i in range(0, len(mnist.test.images)):
  result = sess.run(correct_prediction, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])})
  if not result:
    print('预测的值是：',sess.run(prediction, feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])}))
    print('实际的值是：',sess.run(label,feed_dict={x: np.array([mnist.test.images[i]]), y_: np.array([mnist.test.labels[i]])}))
    one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
    pic_matrix = np.matrix(one_pic_arr, dtype="float")
    plt.imshow(pic_matrix)
    plt.show()