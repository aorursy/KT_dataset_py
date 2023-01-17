import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

style.use('ggplot')

%matplotlib inline



print("Tensorflow Version ", tf.VERSION)

print("Numpy Version ", np.__version__)
np.random.seed(00)

def random_points(num_points):

    vectors_set = []

    for i in range(0, num_points):

        x1 = np.random.normal(0.0, 0.55)

        y1 = 0.1 * x1 + 0.3 + np.random.normal(0.0, 0.03) # add e so that points do not fully corresponds to a line

        vectors_set.append([x1, y1])

    return vectors_set

v_s = random_points(num_points = 1000)



x_data = [v[0] for v in v_s]

y_data = [v[1] for v in v_s]
# plot the variation of x_data

sns.set(color_codes=True)

sns.distplot(x_data)
# plot the x_data and y_data

plt.plot(x_data, y_data, 'ro', label="Original data")

plt.legend(loc="best")

plt.show()
print("We are having input data that is 'x_data' of length ", len(x_data))

print("\n")

print("We are having output data that is 'y_data' of length ", len(y_data))
W = tf.Variable(tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0)) #  -1 < W < 1

b = tf.Variable(tf.zeros(shape=[1]))



y = W * x_data + b # generalized equation
sess = tf.InteractiveSession()

print("Initial value of weight 'w' is ", tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0).eval())
print("Initial value of bias 'b' is ", tf.zeros(shape=[1]).eval())
# create a loss function

# reduce_mean calculates the mean of the elements

loss = tf.reduce_mean(input_tensor = tf.square(y - y_data)) # average((predicted_value - actual_value)^2)
# using optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train = optimizer.minimize(loss = loss)
# initialize the global variables

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

saver = tf.train.Saver(max_to_keep=4) # Save the model



loss_variation = []



for step in range(0, 9): # over 8 iterations

    sess.run(train)

    new_W = sess.run(W)

    new_b = sess.run(b)

    new_loss = sess.run(loss)

    loss_variation.append(new_loss)

    saver.save(sess, './simple_linear_model', global_step=step)

    print(step, new_W, new_b, new_loss)

    

    # Graphic display

    plt.plot(x_data, y_data, 'bo', label="Original data")

    plt.plot(x_data, new_W * x_data + new_b)

    plt.xlabel('X')

    plt.xlim(-2, 2)

    plt.ylabel('Y')

    plt.ylim(0.1, 0.6)

    plt.legend(loc="best")

    plt.show()
plt.plot(loss_variation, label="loss")

plt.xlabel("no. of iterations")

plt.ylabel("loss")

plt.legend(loc="best")

plt.show()
with tf.Session() as session:

    saver = tf.train.import_meta_graph('simple_linear_model-8.meta')

    saver.restore(session, tf.train.latest_checkpoint('./'))

    print("Final value of weight after 8 iterations is ", session.run(W))

    print("Final value of bias after 8 iterations is ", session.run(b))

    print("Final value of loss after 8 iterations is ", session.run(loss))
with tf.Session() as session:

    saver = tf.train.import_meta_graph('simple_linear_model-7.meta')

    saver.restore(session, './simple_linear_model-7')

    print("Final value of weight after 7 iterations is ", session.run(W))

    print("Final value of bias after 7 iterations is ", session.run(b))

    print("Final value of loss after 7 iterations is ", session.run(loss))
with tf.Session() as session:

    saver = tf.train.import_meta_graph('simple_linear_model-6.meta')

    saver.restore(session, './simple_linear_model-6')

    print("Final value of weight after 6 iterations is ", session.run(W))

    print("Final value of bias after 6 iterations is ", session.run(b))

    print("Final value of loss after 6 iterations is ", session.run(loss))