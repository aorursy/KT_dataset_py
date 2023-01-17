import tensorflow.compat.v1 as tf

import numpy as np

import warnings

import matplotlib.pyplot as plt

%matplotlib inline
tf.disable_eager_execution()

warnings.filterwarnings('ignore')
#run time step

time_step = 10

#input size

input_size = 1

#run cell size

cell_size = 32

# learning rate

learning_rate = 0.02
steps = np.linspace(0, np.pi*2, 100, dtype = np.float32)

# the data type is float32 beacuse of converting numpy value to float Tensor



x_np = np.sin(steps)

y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label = 'Target (Cos)')

plt.plot(steps, x_np, 'b-', label = 'Input (Sin)')

plt.tight_layout()

plt.legend(loc = 'best')

plt.show()
tf_x = tf.placeholder(tf.float32, [None, time_step, input_size])     # shape (batch, 5, 1)

tf_y = tf.placeholder(tf.float32, [None, time_step, input_size])     # input y
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units = cell_size)

init_s = rnn_cell.zero_state(batch_size = 1, dtype = tf.float32)



outputs, final_s = tf.nn.dynamic_rnn(rnn_cell,

                                    tf_x,                      # Input

                                    initial_state= init_s,      # The initial hidden Layers

                                    time_major= False)
outs_2D= tf.reshape(outputs, [-1, cell_size])

net_outs2D = tf.layers.dense(outs_2D, input_size)

outs = tf.reshape(net_outs2D, [-1, time_step, input_size])



loss = tf.losses.mean_squared_error(labels = tf_y, predictions=outs)

train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.Session()

sess.run(tf.global_variables_initializer())
plt.figure(1, figsize=(10, 8))

plt.ion()



for step in range(10):

    start = step * np.pi

    end = (step + 1) * np.pi

    

    steps = np.linspace(start, end, time_step)

    

    x = np.sin(steps)[np.newaxis, :, np.newaxis]

    y = np.cos(steps)[np.newaxis, :, np.newaxis]

    

    if 'final_s_' not in globals():

        feed_dict = {tf_x:x, tf_y:y}

    else:

        feed_dict = {tf_x:x, tf_y:y, init_s:final_s_}

        

    _, pred_, final_s_ = sess.run([train_optimizer, outs, final_s], feed_dict)

    

    

    plt.plot(steps, y.flatten(), 'r-')

    plt.plot(steps, pred_.flatten(), 'b-')

    plt.ylim((-1.2, 1.2))

    plt.draw()

    plt.pause(0.05)

    

    

plt.ioff()

plt.tight_layout()

plt.show()