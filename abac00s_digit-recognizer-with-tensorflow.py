import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
raw_dataset = pd.read_csv('../input/train.csv').values

np.random.shuffle(raw_dataset)

all_labels = raw_dataset[:, 0]

all_features = raw_dataset[:, 1:] / 255



num_examples = all_labels.size

a, b = num_examples * 8 // 10, num_examples * 9 // 10



dataset = {

    'train_labels': all_labels[:a],

    'dev_labels': all_labels[a:b],

    'test_labels': all_labels[b:],

    

    'train_features': all_features[:a, :],

    'dev_features': all_features[a:b, :],

    'test_features': all_features[b:, :]

}
def plot_digit(pixels):

    plt.imshow(pixels.reshape(28, 28))

    plt.show()



plot_digit(dataset['test_features'][0, :])
def input_nodes(num_features, num_labels):

    X = tf.placeholder(tf.float32, name = 'X', shape = [None, num_features])

    labels = tf.placeholder(tf.int64, name = 'labels', shape = [None])

    with tf.name_scope('labels_to_Y'):

        Y = tf.one_hot(labels, num_labels, name = 'Y')

    return X, labels, Y
def linear_layer(input_, in_size, out_size, name):

    with tf.variable_scope(name):

        W = tf.get_variable('W', initializer=tf.contrib.layers.xavier_initializer(), shape=[out_size, in_size])

        b = tf.get_variable('b', initializer=tf.zeros_initializer(), shape=[out_size])

    return tf.matmul(input_, W, transpose_b=True) + b
def relu_layer(input_, in_size, out_size, name):

    return tf.nn.relu(linear_layer(input_, in_size, out_size, name))
def forward_prop(X, layers):

    for l in range(1, len(layers) - 1):

        with tf.name_scope('relu_layer' + str(l)):

            X = relu_layer(X, layers[l-1], layers[l], 'weights' + str(l))

    l = len(layers) - 1

    with tf.name_scope('linear_layer'):

        X = linear_layer(X, layers[l-1], layers[l], 'weights' + str(l))

    return X
def cost_function(logits, labels):

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
def logits_to_labels(logits, num_labels):

    return tf.argmax(logits, axis=1)
def accuracy(predictions, labels):

    correct = tf.equal(predictions, labels)

    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    return acc
def model(X_train, labels_train, X_dev, labels_dev, layers=[784, 200, 10], num_epochs=100, learning_rate=0.01,

          writer_dir='tensorboard/model1/default', checkpoint='checkpoints/model.ckpt'):

    tf.reset_default_graph()

    

    X, labels, Y = input_nodes(layers[0], layers[-1])

    with tf.name_scope('forward_prop'):

        logits = forward_prop(X, layers)

    with tf.name_scope('cost'):

        cost = cost_function(logits, Y)

    with tf.name_scope('accuracy'):

        acc = accuracy(logits_to_labels(logits, layers[-1]), labels)

    with tf.name_scope('summaries'):

        tf.summary.scalar('cost', cost)

        tf.summary.scalar('accuracy', acc)

    

    with tf.name_scope('optimizer'):

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    

    with tf.name_scope('summaries'):

        merge = tf.summary.merge_all()

    

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    

    graph_writer = tf.summary.FileWriter(writer_dir + '/graph')

    train_writer = tf.summary.FileWriter(writer_dir + '/train')

    dev_writer = tf.summary.FileWriter(writer_dir + '/dev')

    

    with tf.Session() as sess:

        graph_writer.add_graph(sess.graph)

        sess.run(init)

        for i in range(1, num_epochs + 1):

            summary, _, cost_val = sess.run([merge, train_step, cost], { X: X_train, labels: labels_train })

            train_writer.add_summary(summary, i)

            if i % 10 == 0:

                print('{}. iteration: train cost = {}'.format(i, cost_val))

            

            summary, cost_val = sess.run([merge, cost], { X: X_dev, labels: labels_dev })

            dev_writer.add_summary(summary, i)

            if i % 10 == 0:

                print('dev cost = {}'.format(cost_val))

        saver.save(sess, checkpoint)

    

    graph_writer.close()

    train_writer.close()

    dev_writer.close()
model(dataset['train_features'], dataset['train_labels'], dataset['dev_features'], dataset['dev_labels'],

      num_epochs=200, writer_dir='tensorboard/model1/1')
def generate_submission(checkpoint, layers):

    tf.reset_default_graph()

    challenge = pd.read_csv('../input/test.csv').values

    X = tf.placeholder(tf.float32, [None, layers[0]])

    logits = forward_prop(X, layers)

    pred = logits_to_labels(logits, layers[-1])

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, checkpoint)

        pred = pred.eval({ X: challenge })

    df = pd.DataFrame(data=list(zip(range(1, pred.size+1), pred)), columns=['ImageId', 'Label'])

    return df
df = generate_submission('checkpoints/model.ckpt', [784, 200, 10])

df.to_csv('answer1.csv', index=False, header=True)

df