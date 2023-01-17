# Daniel Balle 2019

import zipfile

import os

import datetime



import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from scipy import ndimage



tf.enable_eager_execution()

tf.set_random_seed(32)
BASE_DIR = '/kaggle/input/aerial-cactus-identification'

IMAGE_PATH = '{}/train/train'.format(BASE_DIR)



train_df = pd.read_csv("{}/train.csv".format(BASE_DIR))
# Function to read some image

def read_image(image_id, base_dir=IMAGE_PATH, transformation=False):

    train_image = mplimg.imread("{}/{}".format(base_dir, image_id)) / 255.0

    if not transformation:

        return train_image



    # Random data augmentation

    train_image = ndimage.rotate(train_image, np.random.choice([0, 1, 2, 3]) * 90, mode='nearest')

    if np.random.rand() > 0.5:

        train_image = np.flip(train_image, np.random.choice([(0), (1), (0, 1)]))  # don't flip colors

    train_image + (np.random.rand() * 0.2) - 0.1  # brightness

    return train_image
# Plot some data

plt.figure(figsize=(8, 8))

for i in range(9):

    plt.subplot(3, 3, i+1)

    plt.xticks([])

    plt.yticks([])

    train_image = read_image(train_df['id'][i], transformation=True)

    plt.imshow(train_image)

    plt.xlabel(train_df['has_cactus'][i])

plt.show()
IMAGE_DIMENSION = read_image(train_df['id'][0]).shape
# Split data into train (80%) and validation set (20%)

shuffled_data = train_df.sample(frac=1)



TRAIN_SIZE = int(len(train_df) * 0.8)

VALIDATION_SIZE = len(train_df) - TRAIN_SIZE



train_set = shuffled_data[0:TRAIN_SIZE]

validation_set = shuffled_data[TRAIN_SIZE:]



print(TRAIN_SIZE)

print(VALIDATION_SIZE)
# check distribution of positives vs negatives

print("Training distribution:\n------")

print("Positives: {}\nNegatives: {}".format(

    list(train_set['has_cactus']).count(1),

    list(train_set['has_cactus']).count(0)))

print(" = {}\n".format(list(train_set['has_cactus']).count(1) / TRAIN_SIZE))



print("Validation distribution:\n------")

print("Positives: {}\nNegatives: {}".format(

    list(validation_set['has_cactus']).count(1),

    list(validation_set['has_cactus']).count(0)))

print(" = {}".format(list(validation_set['has_cactus']).count(1) / VALIDATION_SIZE))
num_samples = list(train_set['has_cactus']).count(1)



# Sample with replacement from the negatives to balance the classes

positive_train_df = train_set[train_set['has_cactus'] == 1]

negative_train_df = train_set[train_set['has_cactus'] == 0].sample(num_samples, replace=True)



balanced_train_set = pd.concat([positive_train_df, negative_train_df], ignore_index=True).sample(frac=1)

BALANCED_TRAIN_SIZE = len(balanced_train_set)
# Generator for test and train data - needs to be callable

def train_gen():

    for _, row in balanced_train_set.iterrows():

        yield (read_image(row['id'], transformation=True), row['has_cactus'])



def validation_gen():

    for _, row in validation_set.iterrows():

        yield (read_image(row['id'], transformation=True), row['has_cactus'])
BATCH_SIZE = 32



train_ds = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float32, tf.int16)).batch(BATCH_SIZE).repeat()

validation_ds = tf.data.Dataset.from_generator(validation_gen, output_types=(tf.float32, tf.int16)).batch(BATCH_SIZE).repeat()
from tensorflow.keras import layers

from tensorflow.keras import regularizers
# Let's try a simple CNN

# TODO(balle) try tf.keras.layers.BatchNormalization

model = tf.keras.Sequential([

    layers.Conv2D(16, (3, 3), activation='relu', input_shape=IMAGE_DIMENSION),

    layers.Conv2D(32, (3, 3), activation='relu'),

    layers.BatchNormalization(),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Dropout(0.5),

    layers.MaxPooling2D((2, 2)),



    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),

    layers.BatchNormalization(),

    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),

    layers.Dropout(0.5),

    layers.MaxPooling2D((2, 2)),



    layers.Flatten(),

    # layers.Dense(32, activation=tf.nn.relu),

    layers.Dense(1, activation=tf.nn.sigmoid)

])



model.summary()



# note: categorical_crossentropy vs. softmax_crossentropy

model.compile(

  optimizer='adam',  # TODO(balle) tune learning rate!

  loss='binary_crossentropy',

  metrics=['accuracy']

)
MAX_STEPS_PER_EPOCH = int(BALANCED_TRAIN_SIZE/BATCH_SIZE)

MAX_VALIDATION_STEPS = int(VALIDATION_SIZE/BATCH_SIZE) * 5  # more validation
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=5)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

logdir = "/tensorboard/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)



history = model.fit_generator(train_ds,

                              epochs=20,

                              validation_data=validation_ds,

                              steps_per_epoch=MAX_STEPS_PER_EPOCH,

                              validation_steps=MAX_VALIDATION_STEPS,

                              callbacks=[early_stopping, checkpointer, tensorboard])
model.load_weights('weights.hdf5')
# Print some predictions

plt.figure(figsize=(8, 8))

for i in range(9):

    j = np.random.randint(0, len(train_df))

    plt.subplot(3, 3, i+1)

    plt.xticks([])

    plt.yticks([])

    train_image = read_image(train_df['id'][j], transformation=False)

    plt.imshow(train_image)

    pred = model.predict(np.array([train_image]))[0][0]

    plt.xlabel("true: {}\npredicted: {:.2f}".format(train_df['has_cactus'][j], pred))

plt.show()
def prediction_gen():

    for _, row in validation_set.iterrows():

        yield read_image(row['id'], transformation=True)



# Predict over multiple transformer images and take mean

prediction_ds = tf.data.Dataset.from_generator(prediction_gen, output_types=(tf.float32)).batch(BATCH_SIZE)  # don't shuffle

multiple_predictions = [

    model.predict_generator(prediction_ds)

    for i in range(10)

]

predictions = np.mean(np.array(multiple_predictions), axis=0)
# Compute accuracy

from sklearn.metrics import accuracy_score

print(accuracy_score(validation_set['has_cactus'], np.round(predictions).astype('int32')))
# Compute AUC

from sklearn.metrics import roc_auc_score

print(roc_auc_score(validation_set['has_cactus'], predictions))
sample_submission = pd.read_csv('{}/sample_submission.csv'.format(BASE_DIR))

sample_submission.head()
test_ids = os.listdir('{}/test/test'.format(BASE_DIR))
def test_gen():

    for test_id in test_ids:

        yield read_image(test_id, base_dir='{}/test/test'.format(BASE_DIR), transformation=True)



test_ds = tf.data.Dataset.from_generator(test_gen, output_types=(tf.float32)).batch(BATCH_SIZE)  # don't shuffle

multiple_test_predictions = [

    model.predict_generator(test_ds)              

    for i in range(10)

]

test_predictions = np.mean(np.array(multiple_test_predictions), axis=0)
submission = pd.DataFrame({'id': test_ids, 'has_cactus': test_predictions.flatten()})

submission.head()
submission.to_csv('submission.csv', index = False, header = True)