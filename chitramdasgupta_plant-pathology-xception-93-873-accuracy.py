import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_style('darkgrid')

import sklearn

import tensorflow as tf

from tensorflow import keras



from kaggle_datasets import KaggleDatasets

import os
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)
AUTOTUNE = tf.data.experimental.AUTOTUNE

GCS_PATH = KaggleDatasets().get_gcs_path()

BATCH_SIZE = 16 * strategy.num_replicas_in_sync  # Perfect batch size for speed and performance

# IMAGE_SIZE = [256, 256]

# IMAGE_RESIZE = [150, 150]



IMAGE_SIZE = 800
def format_path(st):

    return GCS_PATH + '/images/' + st + '.jpg'
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')



train.head()
train_paths = train['image_id'].apply(format_path).values

test_paths = test['image_id'].apply(format_path).values



train_labels = train.loc[:, 'healthy':].values



# print(train_paths[: 3])

# print(train_labels[: 3])
from sklearn.model_selection import train_test_split



train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=0.15, random_state=2020)
def decode_image(image, label=None):

    image = tf.io.read_file(image)

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])



    return image if label is None else (image, label)
def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    return image if label is None else (image, label)
def get_training_dataset():

    return (

        tf.data.Dataset

            .from_tensor_slices((train_paths, train_labels))

            .map(decode_image, num_parallel_calls=AUTOTUNE)

            .cache()

            .map(data_augment, num_parallel_calls=AUTOTUNE)

            .repeat()

            .shuffle(512)

            .batch(BATCH_SIZE)

            .prefetch(AUTOTUNE)

        )



def get_validation_dataset(ordered=False):

    return (

        tf.data.Dataset

            .from_tensor_slices((valid_paths, valid_labels))

            .map(decode_image, num_parallel_calls=AUTOTUNE)

            .cache()

            .batch(BATCH_SIZE)

            .prefetch(AUTOTUNE)

    )



def get_test_dataset(ordered=False):

    return (

        tf.data.Dataset

            .from_tensor_slices(test_paths)

            .map(decode_image, num_parallel_calls=AUTOTUNE)

            .map(data_augment, num_parallel_calls=AUTOTUNE)

            .batch(BATCH_SIZE)

    )
train_dataset = get_training_dataset()

valid_dataset = get_validation_dataset()
def build_model():

    

    base_model = tf.keras.applications.Xception(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')



    base_model.trainable = True

#     set_trainable = False



#     for layer in base_model.layers:

#         if layer.name == 'block13_sepconv1':

#             set_trainable = True

#             layer.trainable = True

#         if set_trainable:

#             layer.trainable = True

#         else:

#             layer.trainable = False

    

    model = keras.models.Sequential([

        base_model,

        keras.layers.Dropout(0.5),

        keras.layers.BatchNormalization(),

        keras.layers.GlobalMaxPooling2D(),

        keras.layers.Dense(4, activation='softmax'),

    ])



    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001), 

                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    

    return model
with strategy.scope():  

    model = build_model()
NUM_TRAINING_IMAGES = 0

NUM_VALIDATION_IMAGES = 0

for temp_path in os.listdir('../input/plant-pathology-2020-fgvc7/images'):

    if temp_path.startswith('Train'):

        NUM_TRAINING_IMAGES += 1

        

NUM_VALIDATION_IMAGES = int(NUM_TRAINING_IMAGES * 0.15)

        

print(NUM_TRAINING_IMAGES)

print(NUM_VALIDATION_IMAGES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

VALID_STEPS = NUM_VALIDATION_IMAGES // BATCH_SIZE
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
history = model.fit(

    train_dataset, epochs=100,

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=valid_dataset,

    validation_steps=VALID_STEPS,

    callbacks=[early_stopping_cb],

    verbose=1

)
epochs = len(history.history['loss'])

epochs
y1 = history.history['loss']

y2 = history.history['val_loss']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['loss', 'val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.tight_layout()
y1 = history.history['categorical_accuracy']

y2 = history.history['val_categorical_accuracy']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['categorical_accuracy', 'val_categorical_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('categorical_accuracy')

plt.tight_layout()
res = model.evaluate(valid_dataset, batch_size=VALID_STEPS)
test_dataset = get_test_dataset()



probs = model.predict(test_dataset, verbose=1)

sub.loc[:, 'healthy':] = probs

sub.to_csv('submission.csv', index=False)

sub.head()