# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow==1.13.1

import os

import numpy as np

# import pandas as pd

import tensorflow as tf



LR = 0.01

BatchSize = 50

EPOCH = 2

print(tf.__version__)
def load_data(filefolder):

    ori_filefolder = filefolder

    filefolder = '../input/ve445-2019-fall-project/'+os.path.join(filefolder,filefolder)

    data = np.load(os.path.abspath(filefolder+ '/names_onehots.npy'), allow_pickle=True).item()

    data = data['onehots']

    if ori_filefolder == 'test':

        label_filename = filefolder + '/output_sample.csv'

#         label = pd.read_csv(os.path.abspath(filefolder + '/output_sample.csv'), sep=',')

    else:

        label_filename = filefolder + '/names_labels.csv'

#         label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.csv'), sep=',')

    label = []

    with open(label_filename,'r') as f:

        header = f.readline().replace('\n','').split(',')

        if header[0]=='Label':

            label_index = 0

        else:label_index = 1

        for line in f.readlines():

            line = line.replace('\n','').split(',')

            label.append(int(line[label_index]))

    label = np.array(label)

#     label = label['Label'].values

    return data, label
# data,label = load_data('test')

# print(data.shape)
def net(onehots_shape):  # [73,398]

    if not isinstance(onehots_shape, list):

        onehots_shape = list(onehots_shape)

    input = tf.placeholder(tf.float32, [None] + onehots_shape, name='input')

    input = tf.reshape(input, [-1] + onehots_shape + [1])

    # input = tf.reshape(input, [None, 73, 398, 1])

    label = tf.placeholder(tf.int32, [None], name='label')

    label = tf.one_hot(label, 2)

    conv1 = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)(input)

    pool1 = tf.keras.layers.MaxPool2D(2, 2)(conv1)

    conv2 = tf.keras.layers.Conv2D(32, 3, (1, 2), padding='same', activation=tf.nn.relu)(pool1)

    pool2 = tf.keras.layers.MaxPool2D(2, 2)(conv2)

    flat = tf.reshape(pool2, [-1, 18*50*32])

    output = tf.keras.layers.Dense(2, name='output')(flat)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=output)  # compute cost

    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    accuracy = tf.metrics.accuracy(  # return (acc, update_op), and create 2 local variables

        labels=tf.argmax(label, axis=1), predictions=tf.argmax(output, axis=1), )[1]

    init_op = tf.group(tf.global_variables_initializer(),

                       tf.local_variables_initializer())  # the local var is for accuracy_op

    return init_op, train_op, loss, accuracy
train_x, train_y = load_data('train')

valid_x, valid_y = load_data('validation')

test_x, test_y = load_data('test')

init_op, train_op, loss, accuracy = net(train_x.shape[1:])

sess = tf.Session()

sess.run(init_op)



train_size = train_x.shape[0]

for epoch in range(EPOCH):

    for i in range(0, train_size, BatchSize):

        b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]

        _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})

    if epoch % 1 == 0:

        accuracy_ = 0

        for i in range(0, valid_x.shape[0], BatchSize):

            b_x, b_y = valid_x[i:i + BatchSize], valid_y[i:i + BatchSize]

            accuracy_ += sess.run(accuracy, {'input:0': b_x, 'label:0': b_y})

        accuracy_ = accuracy_ * BatchSize / valid_x.shape[0]

        print('epoch:', epoch, '| train loss: %.4f' % loss_, '| valid accuracy: %.2f' % accuracy_)

accuracy_ = 0

for i in range(0, test_x.shape[0], BatchSize):

    b_x, b_y = test_x[i:i + BatchSize], test_y[i:i + BatchSize]

    accuracy_ += sess.run(accuracy, {'input:0': b_x, 'label:0': b_y})

accuracy_ = accuracy_ * BatchSize / test_x.shape[0]

print('test accuracy:%.2f' % accuracy_)



saver = tf.train.Saver()

saver.save(sess, './weights/model')

sess.close()
def load_model():

    data = np.load('../input/ve445-2019-fall-project/test/test/names_onehots.npy', allow_pickle=True).item()

    onehots = data['onehots']

    name = data['names']

    data_size = onehots.shape[0]

    

    _, _, loss, accuracy = net(train_x.shape[1:])

    sess = tf.Session()

    saver = tf.train.Saver()

    saver.restore(sess, './weights/model')



    prediction = []

    for i in range(0, data_size, BatchSize):

        test_output = sess.run('output/BiasAdd:0', {'input:0': onehots[i:i + BatchSize]})

        pred = np.argmax(test_output, axis=1)

        prediction.extend(list(pred))

    sess.close()

    f = open('output_5130309315.csv', 'w')

    f.write('Chemical,Label\n')

    for i, v in enumerate(prediction):

        f.write(name[i] + ',%d\n' % v)

    print('predict all')

    f.close()

tf.reset_default_graph()

load_model()
# for node in sess.graph_def.node:

#     print(node)