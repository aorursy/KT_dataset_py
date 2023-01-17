import numpy as np
import plotly
import tensorflow as tf

from plotly.offline import iplot
from plotly import graph_objs as go
# fix seed for reproducibility
# np.random.seed(8787)

# for plotly 
plotly.offline.init_notebook_mode()
def sample_z(m, n):
    return np.random.normal(-1., 1., size=[m, n])
def get_y(x):
    return 10 + x * x
def sample_data(n=10000, scale=100):
    data = []

    x = scale * (np.random.random_sample((n,)) - 0.5)

    for i in range(n):
        y_i = get_y(x[i])
        data.append((x[i], y_i))

    return np.array(data)
data = sample_data()

# show generated data
trace = go.Scatter(
    x=[d[0] for d in data],
    y=[d[1] for d in data],
    mode = 'markers'
)

data = [trace]

iplot(data)

def generator(z, hsize=None, reuse=False):
    if hsize is None:
        hsize = [16, 16]
    with tf.variable_scope('GAN/Generator', reuse=reuse):
        h1 = tf.layers.dense(z, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2, 2)

    return out
def discriminator(x, hsize=None, reuse=False):
    if hsize is None:
        hsize = [16, 16]

    with tf.variable_scope('GAN/Discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, hsize[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.leaky_relu)    
        h3 = tf.layers.dense(h2, 2)    
        out = tf.layers.dense(h3, 1)
    
    return out, h3
# placeholder (input for neural net)
x = tf.placeholder(tf.float32, [None, 2])
z = tf.placeholder(tf.float32, [None, 2])

# generate samples from input z
g_samples = generator(z)

# generated samples feed into discriminator, also the real samples
real_logits, real_representation = discriminator(x)
fake_logits, fake_representation = discriminator(g_samples, reuse=True)

# define loss for generator and discriminator
generator_loss = tf.reduce_mean( 
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits))
)

discriminator_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)) + 
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits))
)



# learn
generator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='GAN/Generator')
discriminator_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='GAN/Discriminator')

generator_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(generator_loss, var_list=generator_variables)
discriminator_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(discriminator_loss, var_list=discriminator_variables)


batch_size = 256

# initialise session
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)


n_discriminator_steps = 5
n_generator_steps = 1

# train in a alternating ways
for i in range(100000):
    x_batch = sample_data(n=batch_size)
    z_batch = sample_z(batch_size, 2)

    for __ in range(n_discriminator_steps):
        _, dloss = sess.run([discriminator_step, discriminator_loss], feed_dict={x: x_batch, z: z_batch})

    for __ in range(n_generator_steps):
        _, gloss = sess.run([generator_step, generator_loss], feed_dict={z: z_batch})

    if i % 5000 == 0:
        print(f'Iterations: {i}\t Discriminator loss: {dloss}\t Generator loss: {gloss}')
# plot generator result
z_batch = sample_z(10000, 2)
g_plot = sess.run(g_samples, feed_dict={z: z_batch})


real_data = sample_data()
trace1 = go.Scatter(
    x=[d[0] for d in g_plot],
    y=[d[1] for d in g_plot],
    mode = 'markers'
)

trace2 = go.Scatter(
    x=[d[0] for d in real_data],
    y=[d[1] for d in real_data],
    mode = 'markers'
)

data = [trace1, trace2]

iplot(data)
