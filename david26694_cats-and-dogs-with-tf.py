import tensorflow as tf
# Common imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

np.random.seed(42)

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
from sklearn.datasets import load_sample_image

#To plot images

def plot_color_image(image):
    plt.imshow(image.astype(np.uint8),interpolation="nearest")
    plt.axis("off")
import matplotlib.image as mpimg
test_image = mpimg.imread(os.path.join("../input/train", "cat.0.jpg"))
plot_color_image(test_image)
animals_root_path = '../input'

animals_classes = ['cat', 'dog']
from collections import defaultdict
import re

image_paths = defaultdict(list)

train_path = os.path.join(animals_root_path, 'train')

for animal in animals_classes:
    for filepath in os.listdir(train_path):
        if filepath.startswith(animal):
            image_paths[animal].append(os.path.join(train_path, filepath))
image_paths['cat'][0]
for paths in image_paths.values():
    paths.sort()    
import matplotlib.image as mpimg

n_examples_per_class = 6
channels = 3

for animal in animals_classes:
    print("Class:", animal)
    plt.figure(figsize=(10,5))
    for index, example_image_path in enumerate(image_paths[animal][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)[:, :, :channels]
        plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis("off")
    plt.show()
from scipy.misc import imresize
from skimage.transform import resize

def prepare_image(image, target_width = 80, target_height = 80, max_zoom = 0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height
        
    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)
    
    # Next, we can select a random location on the image for this bounding box.
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    
    # Let's crop the image using the random bounding box we built.
    image = image[y0:y1, x0:x1]

    # Let's also flip the image horizontally with 50% probability:
    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    # Now, let's resize the image to the target dimensions.
    image = resize(image, (target_width, target_height), mode = 'constant')
    # Finally, let's ensure that the colors are represented as
    # 32-bit floats ranging from 0.0 to 1.0 (for now):
    return image.astype(np.float32) #/ 255
plt.figure(figsize=(6, 8))
plt.imshow(example_image)
plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
plt.axis("off")
plt.show()
rows, cols = 2, 3

plt.figure(figsize=(14, 8))
for row in range(rows):
    for col in range(cols):
        prepared_image = prepare_image(example_image)
        plt.subplot(rows, cols, row * cols + col + 1)
        plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
        plt.imshow(prepared_image)
        plt.axis("off")
plt.show()
height = 80
width = 80
channels = 3
n_inputs = height * width * channels




conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

conv3_fmaps = 128
conv3_ksize = 3
conv3_stride = 1
conv3_pad = "SAME"



n_fc1 = 32

#learn_rate_value = 1e-3

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
    y = tf.placeholder(tf.float32, shape=[None], name="y")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    iters_per_epoch = tf.placeholder(tf.int32, name = 'iters_per_epoch')

global_step = tf.Variable(0)    


def convBlock(X, conv_fmaps, conv_ksize, conv_stride, 
              conv_pad, name_index):
    conv = tf.layers.conv2d(X, filters=conv_fmaps, kernel_size=conv_ksize,
                         strides=conv_stride, padding=conv_pad,
                         activation=tf.nn.relu, name="conv" + str(name_index))
    bn = tf.layers.batch_normalization(conv, name = 'batchnorm' + str(name_index))
    relu = tf.nn.relu(bn, name = 'relu' + str(name_index))
    pool = tf.nn.max_pool(relu, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                           padding = 'VALID', name = 'pool' + str(name_index))
    return pool
    
firstBlock = convBlock(X, conv1_fmaps, conv1_ksize, conv1_stride, conv1_pad, '1')
secondBlock = convBlock(firstBlock, conv2_fmaps, conv2_ksize, conv2_stride, conv2_pad, '2')
pool3 = convBlock(secondBlock, conv3_fmaps, conv3_ksize, conv3_stride, conv3_pad, '3')


with tf.name_scope("fc1"):
    flat_inputs = tf.contrib.layers.flatten(pool3)
    fc1 = tf.layers.dense(flat_inputs, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, units=1, name="output")
    logits = tf.reshape(logits, shape = [-1])
    Y_proba = tf.nn.sigmoid(logits, name="Y_proba")

with tf.name_scope("train"):
    learn_rate = tf.train.cosine_decay_restarts(learning_rate,
                                                global_step,
                                                iters_per_epoch,
                                                alpha = learning_rate/10.,
                                                name= 'LearningRate')
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    training_op = tf.train.AdamOptimizer(learn_rate).minimize(loss,
                                                              global_step=global_step)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_op = optimizer.minimize(loss)

    
with tf.name_scope("learn_rate_finder"):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    training_op_finder = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                       global_step=global_step)


with tf.name_scope("eval"):
    correct_prediction = tf.equal(tf.to_int32(Y_proba > 0.5),
                                  tf.cast(y, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    saver = tf.train.Saver()
animal_class_ids = {animal_class: index for index, animal_class in enumerate(animals_classes)}
animal_class_ids
animals_paths_and_classes = []
for animal, paths in image_paths.items():
    for path in paths:
        animals_paths_and_classes.append((path, animal_class_ids[animal]))
val_ratio = 0.01
train_size = int(len(animals_paths_and_classes) * (1 - val_ratio))

np.random.shuffle(animals_paths_and_classes)

animals_paths_and_classes_train = animals_paths_and_classes[:train_size]
animals_paths_and_classes_val = animals_paths_and_classes[train_size:]
from random import sample

def prepare_batch(animals_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(animals_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch

def prepare_batch_indices(animals_paths_and_classes, indices):
    batch_paths_and_classes = [animals_paths_and_classes[i] for i in indices]
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch
def prepare_batch_total(animals_paths_and_classes):
    batch_paths_and_classes = animals_paths_and_classes
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch

X_val, y_val = prepare_batch_total(animals_paths_and_classes_val)
#This function creates an augmented validation set. It does so by creating a list.
#Every element of the list is a different version of the validation data.

def create_test_data(animals_paths_and_classes_val, tta_len):
    val_data_tta = []
    for i in range(tta_len):
        val_data_tta.append(prepare_batch_total(animals_paths_and_classes_val))
    return val_data_tta
val_data = create_test_data(animals_paths_and_classes_val, 5)
def computeAugmentedProbabilities(Y_proba, val_data):
    probabilities = np.zeros((len(val_data[0][1]), len(val_data)))
    i = 0
    for X_val_tta, y_val_tta in val_data:
        probabilities[:, i] = Y_proba.eval(feed_dict={X: X_val_tta, y: y_val_tta})
        i += 1
    return np.mean(probabilities, axis = 1)
def computeAugmentedAccuracy(Y_proba, val_data):
    probs = computeAugmentedProbabilities(Y_proba, val_data)
    labels = val_data[0][1]
    predictions = probs > .5
    return ((predictions == labels).sum())/len(labels)
def learning_rate_finder(learning_rates, indices, 
                         animals_paths_and_classes_train,
                         iters_per_learn_rate):
    losses = []
    learn_rates_used = []
    initial_losses = []
    X_batch, y_batch = prepare_batch_indices(animals_paths_and_classes_train, indices)
    for learn_rate in (learning_rates):
        learn_rate_value = learn_rate
        with tf.Session() as sess:
            init.run(session = sess)
            initial_loss = loss.eval(feed_dict={X: X_batch, y: y_batch}, session = sess)
            initial_losses.append(initial_loss)
            for iteration in (range(len(learning_rates))):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch,
                                                 learning_rate: learn_rate,
                                                 })
                losses.append(loss.eval(feed_dict={X: X_batch, y: y_batch}))
                learn_rates_used.append(learn_rate)
    return initial_losses, losses, learn_rates_used
indices = np.random.randint(0, len(animals_paths_and_classes_train), 128)
learning_rates = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, .5, 1.]
iters_per_learn_rate = 4
initial_losses, losses, learn_rates = learning_rate_finder(learning_rates, indices, 
                                                           animals_paths_and_classes_train,
                                                           iters_per_learn_rate)
initial_losses
lr_finder = pd.DataFrame({'Improvement': losses - initial_losses[0], 'Learning rate': learn_rates})
lr_finder.groupby(by = 'Learning rate').mean()
means = lr_finder.groupby(by = 'Learning rate').mean()
sorted_means = means.sort_values('Improvement')
learn_rate_value = sorted_means.index[0]
n_epochs = 15
batch_size = 128


train_set_size = len(animals_paths_and_classes_train)

n_iterations_per_epoch = int(len(animals_paths_and_classes_train)/batch_size)


with tf.Session() as sess:
    init.run()
    for epoch in range(0, n_epochs): # This will not be reall epochs, due to time constraints
        for iteration in (range(0, n_iterations_per_epoch)):
            indices = np.random.randint(0, train_set_size, size = batch_size)
            X_batch, y_batch = prepare_batch_indices(animals_paths_and_classes_train,
                                                     indices)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch,
                                             learning_rate: learn_rate_value,
                                             iters_per_epoch: n_iterations_per_epoch})
            #if iteration%20 == 0:
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = computeAugmentedAccuracy(Y_proba, val_data)
        print('Epoch:', epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
    save_path = saver.save(sess, "./cats_dogs_model_v1")
