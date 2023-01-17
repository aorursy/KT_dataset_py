import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
## Batch generator with optional filenames parameter which will also return the filenames of the images
## so that they can be identified
def get_batches(X, y, batch_size, filenames=None, distort=False):
    # Shuffle X,y
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)
    i, h, w, c = X.shape

    # Enumerate indexes by steps of batch_size
    for i in range(0, len(y), batch_size):
        batch_idx = shuffled_idx[i:i + batch_size]
        X_return = X[batch_idx]

        # do random flipping of images
        coin = np.random.binomial(1, 0.5, size=None)
        if coin and distort:
            X_return = X_return[..., ::-1, :]

        if filenames is None:
            yield X_return, y[batch_idx]
        else:
            yield X_return, y[batch_idx], filenames[batch_idx]
            
def _scale_input_data(X, contrast=None, mu=104.1353, scale=255.0):
    # if we are adjusting contrast do that
    if contrast and contrast != 1.0:
        X_adj = tf.image.adjust_contrast(X, contrast)
    else:
        X_adj = X

    # cast to float
    if X_adj.dtype != tf.float32:
        X_adj = tf.cast(X_adj, dtype=tf.float32)

    # center the pixel data
    X_adj = tf.subtract(X_adj, mu, name="centered_input")

    # scale the data
    X_adj = tf.divide(X_adj, scale)

    return X_adj

# Function to do the data augmentation on the GPU instead of the CPU, doing it on the CPU significantly slowed down training
# Taken from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
def augment(images, labels,
            horizontal_flip=False,
            vertical_flip=False,
            augment_labels=False,
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

    # My experiments showed that casting on GPU improves training performance
    if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)

    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

            if augment_labels:
                labels = tf.contrib.image.transform(
                    labels,
                    tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        if mixup > 0:
            beta = tf.distributions.Beta(mixup, mixup)
            lam = beta.sample(batch_size)
            ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
            images = ll * images + (1 - ll) * cshift(images)
            labels = lam * labels + (1 - lam) * cshift(labels)

    return images, labels

## load weights from a checkpoint, excluding any or including specified vars and returning initializer function
def load_weights(model_name, exclude=None, include=None):
    model_path = os.path.join("model", model_name + ".ckpt")

    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude, include=include)
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

    return init_fn

## read data from tfrecords file
def read_and_decode_single_example(filenames, label_type='label_normal', normalize=False, distort=False, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    reader = tf.TFRecordReader()

    if label_type != 'label':
        label_type = 'label_' + label_type

    _, serialized_example = reader.read(filename_queue)
    if label_type != 'label_mask':
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'label_normal': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })

        # extract the data
        label = features[label_type]
        image = tf.decode_raw(features['image'], tf.uint8)

        # reshape and scale the image
        image = tf.reshape(image, [299, 299, 1])

        # random flipping of image
        if distort:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

    else:
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string)
            })

        label = tf.decode_raw(features['label'], tf.uint8)
        image = tf.decode_raw(features['image'], tf.uint8)

        label = tf.cast(label, tf.int32)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        image = tf.reshape(image, [288, 288, 1])
        label = tf.reshape(label, [288, 288, 1])

        # if distort:
        #     image, label = _image_random_flip(image, label)

    if normalize:
        image = tf.image.per_image_standardization(image)

    # return the image and the label
    return image, label

## Load the training data and return a list of the tfrecords file and the size of the dataset
## Multiple data sets have been created for this project, which one to be used can be set with the type argument
def get_training_data(what=10):
    if what == 10:
        train_path_10 = os.path.join("..", "input", "ddsm-mammography", "training10_0", "training10_0.tfrecords")
        train_path_11 = os.path.join("..", "input", "ddsm-mammography", "training10_1","training10_1.tfrecords")
        train_path_12 = os.path.join("..", "input", "ddsm-mammography", "training10_2", "training10_2.tfrecords")
        train_path_13 = os.path.join("..", "input", "ddsm-mammography", "training10_3", "training10_3.tfrecords")

        train_files = [train_path_10, train_path_11, train_path_12, train_path_13]
        total_records = 44712
    else:
        raise ValueError('Invalid dataset!')

    return train_files, total_records

def get_test_data(what=10):
    test_files = os.path.join("..", "input", "ddsm-mammography", "training10_4", "training10_4.tfrecords")
    
    return [test_files], 11178

def _conv2d_batch_norm(input, filters, kernel_size=(3,3), stride=(1,1), training = tf.placeholder(dtype=tf.bool, name="is_training"), epsilon=1e-8, padding="SAME", seed=None, lambd=0.0, name=None, activation="relu"):
    with tf.name_scope('layer_'+name) as scope:
        conv = tf.layers.conv2d(
            input,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=seed),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambd),
            name='conv_'+name
        )

        # apply batch normalization
        conv = tf.layers.batch_normalization(
            conv,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn_'+name
        )

        if activation == "relu":
            # apply relu
            conv = tf.nn.relu(conv, name='relu_'+name)
        elif activation == "elu":
            conv = tf.nn.elu(conv, name="elu_" + name)

    return conv
## ARGUMENTS
epochs = 1
dataset = 10
how = "normal"
action = "eval"
threshold = 0.5
contrast = 1
weight = 7.0
distort = False

batch_size = 32
## Hyperparameters
epsilon = 1e-8

# learning rate
epochs_per_decay = 5
decay_factor = 0.80
staircase = True

# lambdas
lamC = 0.00001
lamF = 0.00250

# use dropout
dropout = True
fcdropout_rate = 0.5
convdropout_rate = 0.001
pooldropout_rate = 0.1

num_classes = 2
train_files, total_records = get_training_data(what=dataset)
test_files, total_records = get_test_data(what=dataset)
print("Number of classes:", num_classes)

steps_per_epoch = int(total_records / batch_size)
print("Steps per epoch:", steps_per_epoch)
##### Build the graph
graph = tf.Graph()

model_name = "model_s2.0.0.36b.10"

with graph.as_default():
    training = tf.placeholder(dtype=tf.bool, name="is_training")
    is_testing = tf.placeholder(dtype=bool, shape=(), name="is_testing")

    # create global step for decaying learning rate
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(0.001,
                                               global_step,
                                               1366,
                                               decay_factor,
                                               staircase=staircase)

    with tf.name_scope('inputs') as scope:
        image, label = read_and_decode_single_example(test_files, label_type=how, normalize=False, distort=False)

        X_def, y_def = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=2000,
                                              seed=None,
                                              min_after_dequeue=1000)

        # Placeholders
        X = tf.placeholder_with_default(X_def, shape=[None, None, None, 1])
        y = tf.placeholder_with_default(y_def, shape=[None])

        # cast to float and scale input data
        X_adj = tf.cast(X, dtype=tf.float32)
        X_adj = _scale_input_data(X_adj, contrast=contrast, mu=127.0, scale=255.0)

        # optional online data augmentation
        if distort:
            X_adj, y = augment(X_adj, y, horizontal_flip=True, vertical_flip=True, mixup=0)

    # Convolutional layer 1
    with tf.name_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            X_adj,
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=100),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv1'
        )

        conv1 = tf.layers.batch_normalization(
            conv1,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn1'
        )

        # apply relu
        conv1_bn_relu = tf.nn.relu(conv1, name='relu1')

    with tf.name_scope('conv1.1') as scope:
        conv11 = tf.layers.conv2d(
            conv1_bn_relu,
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=101),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv1.1'
        )

        conv11 = tf.layers.batch_normalization(
            conv11,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn1.1'
        )

        # apply relu
        conv11 = tf.nn.relu(conv11, name='relu1.1')


    with tf.name_scope('conv1.2') as scope:
        conv12 = tf.layers.conv2d(
            conv11,
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=1101),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv1.2'
        )

        conv12 = tf.layers.batch_normalization(
            conv12,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn1.2'
        )

        # apply relu
        conv12 = tf.nn.relu(conv12, name='relu1.1')

    # Max pooling layer 1
    with tf.name_scope('pool1') as scope:
        pool1 = tf.layers.max_pooling2d(
            conv12,
            pool_size=(3, 3), 
            strides=(2, 2),
            padding='SAME',
            name='pool1'
        )

        # optional dropout
        if dropout:
            pool1 = tf.layers.dropout(pool1, rate=pooldropout_rate, seed=103, training=training)

    # Convolutional layer 2
    with tf.name_scope('conv2.1') as scope:
        conv2 = tf.layers.conv2d(
            pool1,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=104),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2.1'
        )

        conv2 = tf.layers.batch_normalization(
            conv2,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn2.1'
        )

        # apply relu
        conv2 = tf.nn.relu(conv2, name='relu2.1')

    # Convolutional layer 2
    with tf.name_scope('conv2.2') as scope:
        conv22 = tf.layers.conv2d(
            conv2,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=1104),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv2.2'
        )

        conv22 = tf.layers.batch_normalization(
            conv22,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn2.2'
        )

        # apply relu
        conv22 = tf.nn.relu(conv22, name='relu2.2')

    # Max pooling layer 2
    with tf.name_scope('pool2') as scope:
        pool2 = tf.layers.max_pooling2d(
            conv22,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='SAME',
            name='pool2'
        )

        # optional dropout
        if dropout:
            pool2 = tf.layers.dropout(pool2, rate=pooldropout_rate, seed=106, training=training)

    # Convolutional layer 3
    with tf.name_scope('conv3.1') as scope:
        conv3 = tf.layers.conv2d(
            pool2,
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=107),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv3.1'
        )

        conv3 = tf.layers.batch_normalization(
            conv3,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn3.1'
        )

        # apply relu
        conv3 = tf.nn.relu(conv3, name='relu3.1')

    # Convolutional layer 3
    with tf.name_scope('conv3.2') as scope:
        conv32 = tf.layers.conv2d(
            conv3,
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=1107),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv3.2'
        )

        conv32 = tf.layers.batch_normalization(
            conv32,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn3.2'
        )

        # apply relu
        conv32 = tf.nn.relu(conv32, name='relu3.2')

    # Max pooling layer 3
    with tf.name_scope('pool3') as scope:
        pool3 = tf.layers.max_pooling2d(
            conv32,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='SAME',
            name='pool3'
        )

        if dropout:
            pool3 = tf.layers.dropout(pool3, rate=pooldropout_rate, seed=109, training=training)

    # Convolutional layer 4
    with tf.name_scope('conv4') as scope:
            conv4 = tf.layers.conv2d(
                pool3,
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=110),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
                name='conv4'
            )

            conv4 = tf.layers.batch_normalization(
                conv4,
                axis=-1,
                momentum=0.99,
                epsilon=epsilon,
                center=True,
                scale=True,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.ones_initializer(),
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                training=training,
                name='bn4'
            )

            # apply relu
            conv4_bn_relu = tf.nn.relu(conv4, name='relu4')

    # Max pooling layer 4
    with tf.name_scope('pool4') as scope:
            pool4 = tf.layers.max_pooling2d(
                conv4_bn_relu,
                pool_size=(2, 2),
                strides=(2, 2),
                padding='SAME',
                name='pool4'
            )

            if dropout:
                pool4 = tf.layers.dropout(pool4, rate=pooldropout_rate, seed=112, training=training)

    # Convolutional layer 5
    with tf.name_scope('conv5') as scope:
        conv5 = tf.layers.conv2d(
            pool4,
            filters=512,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2, seed=113),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lamC),
            name='conv5'
        )

        conv5 = tf.layers.batch_normalization(
            conv5,
            axis=-1,
            momentum=0.99,
            epsilon=epsilon,
            center=True,
            scale=True,
            beta_initializer=tf.zeros_initializer(),
            gamma_initializer=tf.ones_initializer(),
            moving_mean_initializer=tf.zeros_initializer(),
            moving_variance_initializer=tf.ones_initializer(),
            training=training,
            name='bn5'
        )

        # apply relu
        conv5_bn_relu = tf.nn.relu(conv5, name='relu5')

    # Max pooling layer 4
    with tf.name_scope('pool5') as scope:
        pool5 = tf.layers.max_pooling2d(
            conv5_bn_relu,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='SAME',
            name='pool5'
        )

        if dropout:
            pool5 = tf.layers.dropout(pool5, rate=pooldropout_rate, seed=115, training=training)

    fc1 = _conv2d_batch_norm(pool5, 2048, kernel_size=(5, 5), stride=(5, 5), training=training, epsilon=1e-8,
                             padding="VALID", seed=1013, lambd=lamC, name="fc_1")

    fc2 = _conv2d_batch_norm(fc1, 2048, kernel_size=(1, 1), stride=(1, 1), training=training, epsilon=1e-8,
                             padding="VALID", seed=1014, lambd=lamC, name="fc_2")

    fc3 = tf.layers.dense(
        fc2,
        num_classes,  # One output unit per category
        activation=None,  # No activation function
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=121),
        bias_initializer=tf.zeros_initializer(),
        name="fc_logits"
    )

    logits = tf.squeeze(fc3, name="fc_flat_logits")

    # get the fully connected variables so we can only train them when retraining the network
    fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc")

    with tf.variable_scope('conv1', reuse=True):
        conv_kernels1 = tf.get_variable('kernel')
        kernel_transposed = tf.transpose(conv_kernels1, [3, 0, 1, 2])

    with tf.variable_scope('visualization'):
        tf.summary.image('conv1/filters', kernel_transposed, max_outputs=32, collections=["kernels"])

    # This will weight the positive examples higher so as to improve recall
    weights = tf.multiply(weight, tf.cast(tf.greater(y, 0), tf.float32)) + 1
    mean_ce = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits, weights=weights))

    # Add in l2 loss
    loss = mean_ce + tf.losses.get_regularization_loss()

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss, global_step=global_step)

    # get the probabilites for the classes
    probabilities = tf.nn.softmax(logits, name="probabilities")
    abnormal_probability = 1 - probabilities[:,0]

    # Compute predictions from the probabilities
    if threshold == 0.5:
        predictions = tf.argmax(probabilities, axis=1, output_type=tf.int32)
    else:
        predictions = tf.cast(tf.greater(abnormal_probability, threshold), tf.int32)

    # get the accuracy
    accuracy, acc_op = tf.metrics.accuracy(
        labels=y,
        predictions=predictions,
        updates_collections=tf.GraphKeys.UPDATE_OPS,
        name="accuracy",
    )

    recall, rec_op = tf.metrics.recall(labels=y, predictions=predictions, updates_collections=tf.GraphKeys.UPDATE_OPS, name="recall")
    precision, prec_op = tf.metrics.precision(labels=y, predictions=predictions, updates_collections=tf.GraphKeys.UPDATE_OPS, name="precision")

    f1_score = 2 * ((precision * recall) / (precision + recall))

    # Create summary hooks
    tf.summary.scalar('accuracy', accuracy, collections=["summaries"])
    tf.summary.scalar('cross_entropy', mean_ce, collections=["summaries"])
    tf.summary.scalar('learning_rate', learning_rate, collections=["summaries"])

    # add this so that the batch norm gets run
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Merge all the summaries
    merged = tf.summary.merge_all("summaries")

    print("Graph created...")
## CONFIGURE OPTIONS
init = False
print_every = 5  # how often to print metrics
checkpoint_every = 1  # how often to save model in epochs
print_metrics = True  # whether to print or plot metrics, if False a plot will be created and updated every epoch

config = tf.ConfigProto()
# copy the checkpoints
!mkdir ./model
!cp ../input/fcn-trained-on-ddsm-images/* ./model/
action = "eval"
init = False

## train the model
with tf.Session(graph=graph, config=config) as sess:
    # create the saver
    saver = tf.train.Saver()
    
    # If the model is new initialize variables, else restore the session
    if init:
        sess.run(tf.global_variables_initializer())
        print("Initializing model...")
    else:
        saver.restore(sess, './model/' + model_name + '.ckpt')
        print("Restoring model", model_name)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
        
    # if we are training the model
    if action == "train":

        print("Training model", model_name, "...")

        for epoch in range(epochs):
            sess.run(tf.local_variables_initializer())

            for i in range(steps_per_epoch):
                # create the metadata
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, precision_value, summary, acc_value, cost_value, recall_value = sess.run(
                    [extra_update_ops, prec_op, merged, accuracy, mean_ce, rec_op],
                    feed_dict={
                        training: True,
                    },
                    options=run_options,
                    run_metadata=run_metadata)
                
            # save checkpoint every nth epoch
            if (epoch % checkpoint_every == 0):
                print("Saving checkpoint")
                save_path = saver.save(sess, './model/' + model_name + '.ckpt')

                # Now that model is saved set init to false so we reload it next time
                init = False
    else:
        sess.run(tf.local_variables_initializer())
        
        # evaluate the test data
        for i in range(steps_per_epoch-1):
            valid_acc, valid_recall, valid_precision = sess.run(
                [acc_op, rec_op, prec_op],
                feed_dict={
                    training: False
                })

        # evaluate once more to get the summary
        cv_recall, cv_precision, cv_accuracy = sess.run(
            [recall, precision, accuracy],
            feed_dict={
                training: False
            })

        print("Test Accuracy:", cv_accuracy)
        print("Test Recall:", cv_recall)
        print("Test Precision:", cv_precision)
        
    # stop the coordinator and the threads
    coord.request_stop()
    coord.join(threads)
