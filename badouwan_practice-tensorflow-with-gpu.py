import numpy as np # linear algebra
# 在 [-1,1] 均匀产生100个数
train_X = np.linspace(-1, 1, 100)

# 打乱顺序，人为制造随机性
np.random.shuffle(train_X)

# 拟构造的函数为 Y = 2 * x + 0.3
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
import tensorflow as tf
# 创建模型
with tf.device('/cpu:0'):
    # 占位符
    X = tf.placeholder('float')
    Y = tf.placeholder('float')

    # 模型参数
    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.zeros([1]), name='bias')
    # 前后结构
    z = tf.multiply(X, W) + b

# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 初始化所有变量
init = tf.global_variables_initializer()
# 定义参数
training_epoches = 20
display_step = 2
# 储存训练过程中的数据
fitted_Y = np.array([])
plotdata = {'batchsize': [], 'loss': []}

# 启动session
with tf.Session() as sess:
    sess.run(init)

    # 向模型输入数据
    for epoch in range(training_epoches):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict = {X: x, Y: y})

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict = {X: train_X, Y: train_Y})

            print('Epoch:', epoch + 1, 'cost=', loss, 'W=', sess.run(W), 'b=', sess.run(b))
            if not (loss == 'NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
    
    # 储存拟合数据，供下一步绘图使用
    fitted_Y = sess.run(W) * train_X + sess.run(b)

    print('Finished. cost=', sess.run(cost, feed_dict = {X: train_X, Y: train_Y}), 'W=', sess.run(W), 'b=', sess.run(b))
import matplotlib.pyplot as plt

# 定义函数
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

fig = plt.figure(figsize=(8,8), dpi=90)
fig.add_subplot(211)
plt.plot(train_X, train_Y, 'g.', label='Original data')
plt.plot(train_X, fitted_Y, label='Fitted line')
plt.legend()

plotdata['avgloss'] = moving_average(plotdata['loss'])

fig.add_subplot(212)
plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
plt.show()