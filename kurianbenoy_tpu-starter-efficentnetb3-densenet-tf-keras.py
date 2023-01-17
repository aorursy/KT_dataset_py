from IPython.display import YouTubeVideo

YouTubeVideo("JC84GCU7zqA")
!pip install efficientnet

import efficientnet.tfkeras as efn
import efficientnet.tfkeras as efn

import gc

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import openslide

import os

import tensorflow as tf







from random import randint

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from tqdm import tqdm

%matplotlib inline



print(tf.__version__)

print(tf.keras.__version__)
AUTO = tf.data.experimental.AUTOTUNE

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)





# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path('panda-resized-train-data-512x512')
train_df = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')

print(train_df.shape)

train_df.head()
plt.imshow(plt.imread('/kaggle/input/panda-resized-train-data-512x512/train_images/train_images/'

                      + train_df.iloc[0]['image_id'] + '.png'))
# from https://www.kaggle.com/ateplyuk/panda-tpu-starter-train/

msk = np.random.rand(len(train_df)) < 0.85

train = train_df[msk]

valid = train_df[~msk]
print(train.shape) 

print(valid.shape)

train.head()
train_paths = train["image_id"].apply(lambda x: GCS_DS_PATH + '/train_images/train_images/' + x + '.png').values

valid_paths = valid["image_id"].apply(lambda x: GCS_DS_PATH + '/train_images/train_images/' + x + '.png').values
train_labels = pd.get_dummies(train['isup_grade']).astype('int32').values

valid_labels = pd.get_dummies(valid['isup_grade']).astype('int32').values



print(train_labels.shape) 

print(valid_labels.shape)
BATCH_SIZE= 8 * strategy.num_replicas_in_sync

img_size = 512

EPOCHS = 15

nb_classes = 6
LR_START = 0.00001

LR_MAX = 0.0001 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 3

LR_SUSTAIN_EPOCHS = 1

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .repeat()

    .cache()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

    )
valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)
def get_emodel():

    with strategy.scope():

        en =efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

        en.trainable = True



        model = tf.keras.Sequential([

            en,

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(6, activation='softmax')

        ])

        opt = Adam(learning_rate=1e-3)

        model.compile(optimizer = opt,

            loss = 'categorical_crossentropy',

            metrics=['accuracy']

        )

        print(model.summary())

        return model

    

model = get_emodel()
import os

os.listdir('../input/')
%%time

Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"Enet_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,

       save_weights_only=True,mode='max')

model.load_weights('../input/enetb3prostate/Enet_model.h5')

train_history1 = model.fit(

            train_dataset, 

            validation_data = valid_dataset, 

            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,            

            validation_steps=valid_labels.shape[0] // BATCH_SIZE,            

            callbacks=[lr_callback, Checkpoint],

            epochs=EPOCHS,

            verbose=2

)
def plot_training(H):

	# construct a plot that plots and saves the training history

	with plt.xkcd():

		plt.figure()

		plt.plot(H.history["loss"], label="train_loss")

		plt.plot(H.history["val_loss"], label="val_loss")

		plt.plot(H.history["accuracy"], label="train_acc")

		plt.plot(H.history["val_accuracy"], label="val_acc")

		plt.title("Training Loss and Accuracy")

		plt.xlabel("Epoch #")

		plt.ylabel("Loss/Accuracy")

		plt.legend(loc="lower left")

		plt.show()
plot_training(train_history1)
plt.plot(train_history1.history['accuracy'])

plt.plot(train_history1.history['val_accuracy'])

plt.title('Accuracy throug epochs')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')



plt.show()
def get_densenet_model():

    with strategy.scope():

        rnet = tf.keras.applications.DenseNet201(

                input_shape=(img_size, img_size, 3),

                weights='imagenet',

                include_top=False

            )



        model2 = tf.keras.Sequential([

                rnet,

                tf.keras.layers.GlobalAveragePooling2D(),

                tf.keras.layers.Dense(6, activation='softmax')

            ])



        model2.compile(

            optimizer=tf.keras.optimizers.Adam(lr=0.0001),

            loss = 'categorical_crossentropy',

            metrics=['accuracy']

        )

        model2.summary()

        return model2

    

model2 = get_densenet_model()
%%time

Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"Dnet_basic_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,

       save_weights_only=True,mode='max')

model2.load_weights('../input/enetb3prostate/Dnet_basic_model.h5')

train_history2 = model2.fit(

            train_dataset, 

            validation_data = valid_dataset, 

            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,            

            validation_steps=valid_labels.shape[0] // BATCH_SIZE,            

            callbacks=[lr_callback, Checkpoint],

            epochs=EPOCHS,

            verbose=2

)
plot_training(train_history2)
plt.plot(train_history2.history['accuracy'])

plt.plot(train_history2.history['val_accuracy'])

plt.title('Accuracy throug epochs')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')



plt.show()