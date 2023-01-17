# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install tensorflow==1.15.0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
def load_data(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    data = data['onehots']
    label = pd.read_csv(os.path.abspath(filefolder + '/names_labels.txt'), sep=',')
    label = label['Label'].values
    return data, label
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1_layer = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)
        self.pool1_layer = tf.keras.layers.MaxPool2D(2, 2)
        self.conv2_layer = tf.keras.layers.Conv2D(32, 3, (1, 2), 'same', activation=tf.nn.relu)
        self.pool2_layer = tf.keras.layers.MaxPool2D(2, 2)
        # flat
        self.FCN = tf.keras.layers.Dense(2)
        # softmax

    def call(self, inputs):
        x = self.conv1_layer(inputs)
        x = self.pool1_layer(x)
        x = self.conv2_layer(x)
        x = self.pool2_layer(x)
        flat = tf.reshape(x, [-1, 18*50*32])
        output = self.FCN(flat)
        output_with_sm = tf.nn.softmax(output)
        return output, output_with_sm
# parameters
LR = 0.01
BatchSize = 128
EPOCH = 2

train_data_path = "/kaggle/input/cs410-2020-fall-ai-project-1/data/train/"
validation_data_path = "/kaggle/input/cs410-2020-fall-ai-project-1/data/validation/"
# data
train_x, train_y = load_data(train_data_path)
valid_x, valid_y = load_data(validation_data_path)


# model & input and output of model
model = MyModel()

onehots_shape = list(train_x.shape[1:])
input_place_holder = tf.placeholder(tf.float32, [None] + onehots_shape, name='input')
input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + onehots_shape + [1])
label_place_holder = tf.placeholder(tf.int32, [None], name='label')
label_place_holder_2d = tf.one_hot(label_place_holder, 2)
output, output_with_sm = model(input_place_holder_reshaped)
model.summary()  # show model's structure

# loss
bce = tf.keras.losses.BinaryCrossentropy()  # compute cost
loss = bce(label_place_holder_2d, output_with_sm)

# Optimizer
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# auc
prediction_place_holder = tf.placeholder(tf.float64, [None], name='pred')
auc, update_op = tf.metrics.auc(labels=label_place_holder, predictions=prediction_place_holder)

# run
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)

    saver = tf.train.Saver()

    train_size = train_x.shape[0]
    best_val_auc = 0
    for epoch in range(EPOCH):
        for i in range(0, train_size, BatchSize):
            b_x, b_y = train_x[i:i + BatchSize], train_y[i:i + BatchSize]
            _, loss_ = sess.run([train_op, loss], {'input:0': b_x, 'label:0': b_y})

            print("Epoch {}: [{}/{}], training set loss: {:.4}".format(epoch, i, train_size, loss_))

        if epoch % 1 == 0:
            val_prediction = sess.run(output_with_sm, {'input:0': valid_x})
            val_prediction = val_prediction[:, 1]
            auc_value = sess.run(update_op, feed_dict={prediction_place_holder: val_prediction, label_place_holder: valid_y})
            print("auc_value", auc_value)
            if auc_value > best_val_auc:
                saver.save(sess, '/kaggle/working/weights/model')
                
def load_test_data_name(filefolder):
    data = np.load(os.path.abspath(filefolder + '/names_onehots.npy'), allow_pickle=True).item()
    onehots = data['onehots']
    name = data['names']
    return onehots, name
# data
test_path = "/kaggle/input/cs410-2020-fall-ai-project-1/data/test/"
test_data, test_name = load_test_data_name(test_path)
name = test_name

# model
tf.reset_default_graph()  # 
model = MyModel()
input_place_holder = tf.placeholder(tf.float32, [None] + list(test_data.shape[1:]), name='input')
input_place_holder_reshaped = tf.reshape(input_place_holder, [-1] + list(test_data.shape[1:]) + [1])
output, output_with_sm = model(input_place_holder_reshaped)

# Predict on the test set
data_size = test_data.shape[0]
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, os.path.abspath('/kaggle/working/weights/model'))
    # saver.restore(sess, os.path.abspath('fds'))
    prediction = []
    for i in range(0, data_size, BatchSize):
        print(i)
        test_output = sess.run(output, {input_place_holder: test_data[i:i + BatchSize]})
        test_output_with_sm = sess.run(output_with_sm, {input_place_holder: test_data[i:i + BatchSize]})
        pred = test_output_with_sm[:, 1]
        prediction.extend(list(pred))
sess.close()
f = open('output_518000001.txt', 'w')
f.write('Chemical,Label\n')
for i, v in enumerate(prediction):
    f.write(name[i] + ',%f\n' % v)
f.close()