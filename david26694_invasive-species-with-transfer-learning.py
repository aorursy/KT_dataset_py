# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
INCEPTION_PATH = os.path.join("../input", "v3-inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")
labels = pd.read_csv('../input/invasive-species-monitoring/train_labels.csv')
labels_invasive = labels[labels['invasive'] == 1]['name']
labels_autoctonous = labels[labels['invasive'] == 0]['name']
image_paths = {}
image_paths['autoctonous'] = ['../input//invasive-species-monitoring/train/' + str(label) + '.jpg' for label in (labels_autoctonous.values)] 
image_paths['invasive'] = ['../input//invasive-species-monitoring/train/' + str(label) + '.jpg' for label in (labels_invasive.values)] 
import matplotlib.image as mpimg

n_examples_per_class = 6
channels = 3

species_classes = ['autoctonous', 'invasive']

for species in species_classes:
    print("Class:", species)
    plt.figure(figsize=(10,5))
    for index, example_image_path in enumerate(image_paths[species][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)[:, :, :channels]
        plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis("off")
    plt.show()
from scipy.misc import imresize
from skimage.transform import resize

def prepare_image(image, target_width = 299, target_height = 299, max_zoom = 0.2):
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
channels = 3
height = 299
width = 299
example_image_path = image_paths['invasive'][0]
example_image = mpimg.imread(example_image_path)[:, :, :channels]
reset_graph()

rows = 2
cols = 3

plt.figure(figsize=(14, 8))
for row in range(rows):
    for col in range(cols):
        prepared_image = prepare_image(example_image)
        plt.subplot(rows, cols, row * cols + col + 1)
        plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
        plt.imshow(prepared_image)
        plt.axis("off")
plt.show()
species_class_ids = {species_class: index for index, species_class in enumerate(species_classes)}
species_class_ids
species_paths_and_classes = []
for species, paths in image_paths.items():
    for path in paths:
        species_paths_and_classes.append((path, species_class_ids[species]))
val_ratio = 0.1
train_size = int(len(species_paths_and_classes) * (1 - val_ratio))

np.random.shuffle(species_paths_and_classes)

species_paths_and_classes_train = species_paths_and_classes[:train_size]
species_paths_and_classes_val = species_paths_and_classes[train_size:]
from random import sample

def prepare_batch(species_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(species_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch

def prepare_batch_indices(species_paths_and_classes, indices):
    batch_paths_and_classes = [species_paths_and_classes[i] for i in indices]
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch
def prepare_batch_total(species_paths_and_classes):
    batch_paths_and_classes = species_paths_and_classes
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch

X_val, y_val = prepare_batch_total(species_paths_and_classes_val)
#This function creates an augmented validation set. It does so by creating a list.
#Every element of the list is a different version of the validation data.

def create_test_data(species_paths_and_classes_val, tta_len):
    val_data_tta = []
    for i in range(tta_len):
        val_data_tta.append(prepare_batch_total(species_paths_and_classes_val))
    return val_data_tta
val_data = create_test_data(species_paths_and_classes_val, 5)
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
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
training = tf.placeholder_with_default(False, shape=[])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

inception_saver = tf.train.Saver()
prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])
n_outputs_1 = 100
#n_outputs_2 = 10
n_outputs_3 = 1
with tf.name_scope("new_output_layer"):
    species_fc1 = tf.layers.dense(prelogits, n_outputs_1, name="species_fc1", activation=tf.nn.relu)
    #species_fc2 = tf.layers.dense(species_fc1, n_outputs_2, name="species_fc2", activation=tf.nn.relu)
    species_logits = tf.layers.dense(species_fc1, n_outputs_3, name="species_logits")
    reshaped_logits = tf.reshape(species_logits, shape = [-1])
    Y_proba = tf.nn.sigmoid(reshaped_logits, name="Y_proba")
y = tf.placeholder(tf.float32, shape=[None])

with tf.name_scope("train"):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=reshaped_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    species_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="species_logits")
    training_op = optimizer.minimize(loss, var_list=species_vars)

with tf.name_scope("eval"):
    correct_prediction = tf.equal(tf.to_int32(Y_proba > 0.5),
                                  tf.cast(y, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 
n_epochs = 10
batch_size = 64
n_iterations_per_epoch = len(species_paths_and_classes_train) // batch_size

with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)

    for epoch in range(n_epochs):
        print("Epoch", epoch, end="")
        for iteration in tqdm(range(n_iterations_per_epoch)):
            #print(".", end="")
            X_batch, y_batch = prepare_batch(species_paths_and_classes_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration % 10 == 0:
                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                print("Train accuracy: ", acc_train)
        acc_val = computeAugmentedAccuracy(Y_proba, val_data)
        print('Epoch:', epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
        save_path = saver.save(sess, "./my_species_model")
