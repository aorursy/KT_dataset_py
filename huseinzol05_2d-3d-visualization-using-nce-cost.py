import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import seaborn as sns

sns.set()
df = pd.read_json('../input/train.json')

df.inc_angle = df.inc_angle.replace('na', 0)

df.inc_angle = df.inc_angle.astype(float).fillna(0.0)
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])

X_train = np.concatenate([x_band1[:, :, :, np.newaxis]

                          , x_band2[:, :, :, np.newaxis]

                         , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)

X_angle_train = np.array(df.inc_angle)

y_train = np.array(df["is_iceberg"])
# just take 200 dataset to do visualization

# we assume this 200 able to generalize the whole dataset

# if not enough, increase the number

X_train = X_train[:500]

y_train = y_train[:500].reshape((-1, 1))

X_angle_train = X_angle_train[:500].reshape((-1, 1))

learning_rate = 0.001

boundary = [-1, 1]

batch_size = 20

dimension_size = 300

epoch = 30
class Model:



    def __init__(self, vocabulary_size):

        self.X = tf.placeholder('float', [None, 75, 75, 3])

        self.X_angle = tf.placeholder('float', (None, 1))

        self.Y = tf.placeholder('float', [None, 1])



        def conv_layer(x, conv, out_shape, name, stride = 1):

            w = tf.Variable(tf.truncated_normal([conv, conv, int(x.shape[3]), out_shape]), name = name + '_w')

            b = tf.Variable(tf.truncated_normal([out_shape], stddev = 0.01), name = name + '_b')

            return tf.nn.conv2d(x, w, [1, stride, stride, 1], padding = 'SAME') + b



        def pooling(x, k = 2, stride = 2):

            return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding = 'SAME')



        with tf.name_scope("conv5-16"):

            conv1 = tf.nn.sigmoid(conv_layer(self.X, 5, 16, '16'))



        with tf.name_scope("maxpool-1"):

            pooling1 = pooling(conv1)

            

        with tf.name_scope("conv5-32"):

            conv2 = tf.nn.sigmoid(conv_layer(pooling1, 5, 32, '32'))

    

        with tf.name_scope("maxpool-2"):

            pooling2 = pooling(conv2)



        with tf.name_scope("conv5-64"):

            conv3 = tf.nn.sigmoid(conv_layer(pooling2, 5, 64, '64'))



        with tf.name_scope("maxpool-3"):

            pooling3 = pooling(conv3)



        with tf.name_scope("conv5-128"):

            conv4 = tf.nn.sigmoid(conv_layer(pooling3, 5, 128, '128'))



        with tf.name_scope("maxpool-4"):

            pooling4 = pooling(conv4)

            

        with tf.name_scope("conv5-256"):

            conv5 = tf.nn.sigmoid(conv_layer(pooling4, 5, 256, '256'))



        with tf.name_scope("maxpool-5"):

            pooling5 = pooling(conv5)



        output_shape = int(pooling5.shape[1]) * int(pooling5.shape[2]) * int(pooling5.shape[3])

        pooling5 = tf.reshape(pooling5, [-1, output_shape])

        pooling5 = tf.concat([pooling5, self.X_angle], axis = 1)

        embeddings = tf.Variable(tf.random_uniform([output_shape + 1, dimension_size], boundary[0], boundary[1]))

        embeddings = tf.matmul(pooling5, embeddings)

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, dimension_size], stddev = 1.0 / np.sqrt(dimension_size)))

        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases, labels = self.Y,

                                                  inputs = embeddings, num_sampled = batch_size, num_classes = vocabulary_size))



        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))

        self.normalized_embeddings = embeddings / norm
sess = tf.InteractiveSession()

model = Model(X_train.shape[0])

sess.run(tf.global_variables_initializer())

for i in range(epoch):

    total_loss = 0

    for k in range(0, (X_train.shape[0] // batch_size) * batch_size, batch_size):

        loss, _ = sess.run([model.loss, model.optimizer], feed_dict = {model.X: X_train[k: k + batch_size, :, :, :], 

                                                                       model.X_angle: X_angle_train[k: k + batch_size, :],

                                                                       model.Y: y_train[k: k + batch_size, :]})

    total_loss += loss

    print('epoch: ', i, 'avg loss: ', total_loss / (X_train.shape[0] // batch_size))
vector_out = sess.run(model.normalized_embeddings, feed_dict = {model.X: X_train, model.X_angle: X_angle_train})
from sklearn.manifold import TSNE

embed_2d = TSNE(n_components = 2).fit_transform(vector_out)

embed_3d = TSNE(n_components = 3).fit_transform(vector_out)
plt.figure(figsize=(8, 5))

label = ['ship', 'ice']

colors = sns.color_palette(n_colors = len(label))

y_train_reshape = y_train.reshape([-1])

for no, _ in enumerate(np.unique(y_train_reshape)):

    plt.scatter(embed_2d[y_train_reshape == no, 0], embed_2d[y_train_reshape == no, 1], c = colors[no], label = label[no])

plt.legend()

plt.show()
data_graph = []

from ast import literal_eval

# i love these colors, dont judge me

colors = ['rgb(0,31,63)', 'rgb(255,133,27)']

for no, _ in enumerate(np.unique(y_train_reshape)):

    graph = go.Scatter3d(

    x = embed_3d[y_train_reshape == no, 0],

    y = embed_3d[y_train_reshape == no, 1],

    z = embed_3d[y_train_reshape == no, 2],

    name = label[no],

    mode = 'markers',

    marker = dict(

        size = 12,

        line = dict(

            color = '#%02x%02x%02x' % literal_eval(colors[no][3:]),

            width = 0.5

            ),

        opacity = 0.5

        )

    )

    data_graph.append(graph)

    

layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.7,

            y = 0.7,

            z = 0.7

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = data_graph, layout = layout)

py.iplot(fig, filename = '3d-scatter')