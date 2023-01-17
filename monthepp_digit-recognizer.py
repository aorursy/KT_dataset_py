# import python standard library

import time



# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

import matplotlib.pyplot as plt



# import tensorflow model class

import tensorflow as tf



# import sklearn model selection

from sklearn.model_selection import train_test_split
# acquiring training and testing data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# visualize head of the training data

df_train.head(n=5)
# visualize tail of the testing data

df_test.tail(n=5)
# combine training and testing dataframe

df_train['datatype'], df_test['datatype'] = 'training', 'testing'

df_train['imageid'] = df_train.index + 1

df_test.insert(0, 'label', np.nan)

df_test['imageid'] = df_test.index + 1

df_data = pd.concat([df_train, df_test], ignore_index=True)
# data dimensions

img_size = 28

num_channels = 1

num_classes = 10



# flat dimensions

img_size_flat = img_size * img_size * num_channels
def imageplot(image: list, label: list, size: tuple, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return an image plot applied for an image data in grayscale picture (m, n) format, RGB picture (m, n, 3) format and RGBA picture (m, n, 4) format.

    

    Args:

        image (list): The image data.

        label (list): The label of an image data.

        size (tuple): The tuple of an image size.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    if nrows is None: nrows = (len(label) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [axes[i].imshow(image[i].reshape(size), interpolation='spline16') for i in range(len(label))]

    return fig
# describe training and testing data

df_data.describe(include='all')
# feature exploration: number 1

number = 1

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 2

number = 2

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 3

number = 3

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 4

number = 4

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 5

number = 5

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 6

number = 6

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 7

number = 7

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 8

number = 8

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 9

number = 9

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature exploration: number 0

number = 0

pixel = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), df_data.columns[1:-2]].values

label = df_data.loc[(df_data['datatype'] == 'training') & (df_data['label'] == number), 'label'].values

_ = imageplot(pixel[:15], label[:15], (img_size, img_size))
# feature extraction: normalize pixel between 0 to 1

col_pixels = df_data.columns[1:-2]

df_data[col_pixels] = df_data[col_pixels] / 255.0
# feature extraction: label

df_data['label'] = df_data['label'].fillna(-1).astype(int)
# convert category codes for data dataframe

df_data = pd.get_dummies(df_data, columns=['datatype', 'label'], drop_first=True)
# describe data dataframe

df_data.describe(include='all')
# verify dtypes object

df_data.info()
# select all features

x = df_data[df_data['datatype_training'] == 1].drop(['imageid', 'datatype_training', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9'], axis=1)

y = df_data.loc[df_data['datatype_training'] == 1, ['label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9']]
# perform train-test (validate) split

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.25, random_state=58)
# weight variable

def weight_variable(shape, name = None, stddev = 0.1):

    return tf.Variable(tf.truncated_normal(shape, stddev=stddev, seed=58), name=name)
# bias variable

def bias_variable(shape, name = None, value = 0.1):

    return tf.Variable(tf.constant(value, shape=shape), name=name)
# conv2d layer

def conv2d_layer(x, weight, bias, padding = 'SAME', strides = [1, 1, 1, 1]):

    return tf.nn.conv2d(x, weight, strides, padding) + bias
# fully connected layer

def fc_layer(x, weight, bias):

    return tf.matmul(x, weight) + bias
# flatten layer

def flatten_layer(x):

    num_features = x.get_shape()[1:4].num_elements()

    return tf.reshape(x, [-1, num_features]), num_features
# max pool layer

def max_pool_layer(x, ksize = [1, 2, 2, 1], padding = 'SAME', strides = [1, 2, 2, 1]):

    return tf.nn.max_pool(x, ksize, strides, padding)
# relu layer

def relu_layer(x):

    return tf.nn.relu(x)
# reset default graph

tf.reset_default_graph()
# placeholder variables used for inputting data to the graph

x_flat = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x_flat')

x_image = tf.reshape(x_flat, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_class = tf.argmax(y_true, axis=1)
# convolution + relu + max pool layer 1

weight_conv1 = weight_variable([5, 5, num_channels, 16], name='weight_conv1')

bias_conv1 = bias_variable([16], name='bias_conv1')

layer_conv1 = max_pool_layer(relu_layer(conv2d_layer(x_image, weight_conv1, bias_conv1)))
# convolution + relu + max pool layer 2

weight_conv2 = weight_variable([5, 5, 16, 64], name='weight_conv2')

bias_conv2 = bias_variable([64], name='bias_conv2')

layer_conv2 = max_pool_layer(relu_layer(conv2d_layer(layer_conv1, weight_conv2, bias_conv2)))
# flatten layer

layer_flat, num_features = flatten_layer(layer_conv2)
# fully connected + relu layer 1

weight_fc1 = weight_variable([num_features, 128], name='weight_fc1')

bias_fc1 = bias_variable([128], name='bias_fc1')

layer_fc1 = relu_layer(fc_layer(layer_flat, weight_fc1, bias_fc1))
# fully connected + relu layer 2

weight_fc2 = weight_variable([128, num_classes], name='weight_fc2')

bias_fc2 = bias_variable([num_classes], name='bias_fc2')

layer_fc2 = relu_layer(fc_layer(layer_fc1, weight_fc2, bias_fc2))
# predicted class label

y_pred_proba = tf.nn.softmax(layer_fc2)

y_pred_class = tf.argmax(y_pred_proba, axis=1)
# cost function

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=layer_fc2))
# optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
# performance metrics

accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_class, y_true_class), tf.float32))
# counter for total number of iterations performed so far

total_epoch = 0

rndobj = np.random.RandomState(seed=58)



def optimize(num_epoch, printcost = True, printfrequency = 1000, train_batch_size = 256, validate_batch_size = 256):

    global total_epoch

    

    # record start time

    timestart = time.time()

    

    for i in range(total_epoch, total_epoch + num_epoch):

        # specify batch size

        train_index = rndobj.choice(x_train.index, replace=True, size=train_batch_size)

        validate_index = rndobj.choice(x_validate.index, replace=True, size=validate_batch_size)

        

        # tensorflow model fit

        feed_dict_train = {x_flat: x_train.loc[train_index], y_true: y_train.loc[train_index]}

        feed_dict_validate = {x_flat: x_validate.loc[validate_index], y_true: y_validate.loc[validate_index]}

        session.run(optimizer, feed_dict=feed_dict_train)

        

        # print status every 1000 iterations

        if printcost and i % printfrequency == 0: print('epoch: %d, training accuracy: %f, testing accuracy: %f' %(i + 1, session.run(accuracy, feed_dict=feed_dict_train), session.run(accuracy, feed_dict=feed_dict_validate)))

    

    # update the total epoch

    total_epoch += num_epoch

    

    # record end time

    timeend = time.time()

    

    # time elapsed

    timeelapsed = timeend - timestart

    

    # print the time elapsed

    print("elapsed time: %f" %timeelapsed)
# create tensorflow session

session = tf.Session()

session.run(tf.global_variables_initializer())
# tensorflow model fit 1 epoch

optimize(1)
# tensorflow model fit 10000 epoch

optimize(10000, printfrequency=1000)
# prepare testing data and compute the observed value

x_test = df_data[df_data['datatype_training'] == 0].drop(['imageid', 'datatype_training', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9'], axis=1)

y_test = pd.DataFrame(y_pred_class.eval(session=session, feed_dict={x_flat: x_test}), columns=['Label'], index=df_data.loc[df_data['datatype_training'] == 0, 'imageid'])
# submit the results

out = pd.DataFrame({'ImageId': y_test.index, 'Label': y_test['Label']})

out.to_csv('submission.csv', index=False)