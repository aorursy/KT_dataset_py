import tensorflow as tf
tf.enable_eager_execution()
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
print(C)
'''
with tf.Session() as sess:
    print(sess.run(C))
'''
x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.))
with tf.GradientTape() as tape: # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
    y_grad = tape.gradient(y, x) # 计算 y 关于 x 的导数
    print([y.numpy(), y_grad.numpy()])

X = tf.constant([[1.0,2.0],[3.0,4.0]])
y = tf.constant([[1.0],[2.0]])

with tf.GradientTape() as tape:
    w = tf.get_variable('w', shape=[2, 1], initializer=tf.constant_initializer([[1.], [2.]]))
    b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer([1.]))
    with tf.GradientTape() as tape:
        L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
        w_grad,b_grad = tape.gradient(L,[w,b])
        print(w_grad)
        print("-"*80)
        print(b_grad)
a = tf.get_variable("a",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer())
b = tf.get_variable("a",shape=[],dtype=tf.float32,initializer=tf.zeros_initializer())

variables = [a, b]
num_epoch = 10000
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
for i in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables)
    #print(grads)
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
print("loss:",loss)









