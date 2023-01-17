# Import library

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

tqdm.pandas()

import os

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

!pip install efficientnet pandarallel

import efficientnet.tfkeras as efn 

import tensorflow as tf

import tensorflow.keras as keras

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
USE_TPU = 'TPU_NAME' in os.environ

if USE_TPU:

    # detect and init the TPU

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)



    # instantiate a distribution strategy

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    tf.compat.v1.enable_eager_execution()

else:

    strategy = tf.distribute.MirroredStrategy()
# Read data

IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"

TEST_PATH = "../input/plant-pathology-2020-fgvc7/test.csv"

TRAIN_PATH = "../input/plant-pathology-2020-fgvc7/train.csv"

SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"



sub = pd.read_csv(SUB_PATH)

test_data = pd.read_csv(TEST_PATH)

train_data = pd.read_csv(TRAIN_PATH)
def init_grabcut_mask(h, w):

    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD

    mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD

    mask[2*h//5:3*h//5, 2*w//5:3*w//5] = cv2.GC_FGD

    #mask[h//2, w//2] = cv2.GC_FGD

    return mask





def remove_background(image, h=136, w=205):

    orig_image = image

    image = cv2.resize(image, (w, h))

    mask = init_grabcut_mask(h, w)

    bgm = np.zeros((1, 65), np.float64)

    fgm = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)

    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    h, w = orig_image.shape[:2]

    mask_binary = cv2.resize(mask_binary, (w, h))

    result = cv2.bitwise_and(orig_image, orig_image, mask=mask_binary)

    return result
def rotate(x: tf.Tensor) -> tf.Tensor:

    """Rotation augmentation



    Args:

        x: Image



    Returns:

        Augmented image

    """

    shape = tf.shape(x)[:-1]



    # Rotate 0, 90, 180, 270 degrees

    x = tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    return tf.image.resize(x, shape)





def flip(x: tf.Tensor) -> tf.Tensor:

    """Flip augmentation



    Args:

        x: Image to flip



    Returns:

        Augmented image

    """

    x = tf.image.random_flip_left_right(x)

    x = tf.image.random_flip_up_down(x)



    return x





def color(x: tf.Tensor) -> tf.Tensor:

    """Color augmentation



    Args:

        x: Image



    Returns:

        Augmented image

    """

    x = tf.image.random_hue(x, 0.08)

    x = tf.image.random_saturation(x, 0.6, 1.6)

    x = tf.image.random_brightness(x, 0.05)

    x = tf.image.random_contrast(x, 0.7, 1.3)

    return x





def zoom(x: tf.Tensor) -> tf.Tensor:

    """Zoom augmentation



    Args:

        x: Image



    Returns:

        Augmented image

    """

    shape = tf.shape(x)[:-1]



    # Generate 20 crop settings, ranging from a 1% to 20% crop.

    scales = list(np.arange(0.8, 1.0, 0.01))

    boxes = np.zeros((len(scales), 4))



    for i, scale in enumerate(scales):

        x1 = y1 = 0.5 - (0.5 * scale)

        x2 = y2 = 0.5 + (0.5 * scale)

        boxes[i] = [x1, y1, x2, y2]



    def random_crop(img):

        # Create different crops for an image

        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(408, 615))

        # Return a random crop

        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]





    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)



    # Only apply cropping 50% of the time

    x = tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

    

    return x
def get_data_generators(preprocess=True, augment=True, IMAGE_SIZE=(408, 615), nfolds=5):

    

    def load_image(image_id):

        file_path = image_id + ".jpg"

        image = cv2.imread(IMAGE_PATH + file_path)

        image = cv2.resize(image, IMAGE_SIZE[::-1])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if preprocess:

            image = remove_background(image)

        return image



    print("Preprocessing training images...")

    train_images = np.stack(train_data["image_id"].parallel_apply(load_image))

    plt.imshow(train_images[0])

    labels = train_data[['healthy', 'multiple_diseases', 'rust', 'scab']]

    dataset = tf.data.Dataset.from_tensor_slices((train_images, np.stack(labels.values)))

    

    def map_func(image, label):

        image = tf.cast(image, tf.float32) / 255

        label = (tf.cast(label, tf.float32) + 0.01) / 1.04

        return image, label

    dataset = dataset.map(map_func, 

                          num_parallel_calls=tf.data.experimental.AUTOTUNE, 

                          deterministic=True)



    if augment:

        # Add augmentations

        augmentations = [flip, color, rotate, zoom]

    

        # Add the augmentations to the dataset

        for f in augmentations:

            # Apply the augmentation, run 4 jobs in parallel.

            dataset = dataset.map(lambda x, y: (f(x), y),

                                  num_parallel_calls=tf.data.experimental.AUTOTUNE, 

                                  deterministic=True)



        # Make sure that the values are still in [0, 1]

        dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y), 

                              num_parallel_calls=tf.data.experimental.AUTOTUNE, 

                              deterministic=True)

    

    fold_size = len(train_data) // nfolds

    folds = []

    for idx in range(nfolds):

        fold = dataset.take(fold_size)

        folds.append(fold)

        dataset = dataset.skip(fold_size)

    

    print("Preprocessing test images...")

    test_images = np.stack(test_data["image_id"].parallel_apply(load_image))

    test = tf.data.Dataset.from_tensor_slices((np.stack(test_images,)))

    

    test = test.map(lambda image: tf.cast(image, tf.float32) / 255, 

                    num_parallel_calls=tf.data.experimental.AUTOTUNE, 

                    deterministic=False)

    test = test.batch(32).prefetch(2)

    

    return folds, test
folds, test = get_data_generators(False, True)
def get_model(): 

    model = keras.Sequential()

    model.add(efn.EfficientNetB7(

        include_top=False, weights='imagenet', input_tensor=None, input_shape=None,

        pooling=None, classes=4))

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dense(128, activation="relu"))

    model.add(keras.layers.Dense(64, activation="relu"))

    model.add(keras.layers.Dense(4, activation="softmax"))

    model.summary()

    return model
with strategy.scope():

    model = get_model()

    model.compile(

        optimizer=keras.optimizers.Adam(learning_rate=1e-3),

        loss=keras.losses.categorical_crossentropy,

        metrics=[keras.metrics.categorical_accuracy])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback



callbacks = [

    EarlyStopping(monitor='val_loss', mode='min', patience=10),

    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.000001),

    CSVLogger('log.csv', append=True, separator=';')

    #TensorBoard(log_dir="tensorboard/")

]



def get_train_val_split(val_fold):

    train_folds = folds[:val_fold] + folds[val_fold + 1:]

    train = train_folds[0]

    for fold in train_folds[1:]:

        train = train.concatenate(fold)

    val = folds[val_fold]

    return train, val



splits = [get_train_val_split(i) for i in range(len(folds))]

train, val = splits[0]

for t, v in splits[1:]:

    train = train.concatenate(t)

    val = val.concatenate(v)

train = train.repeat().batch(32).prefetch(2)

val = val.repeat().batch(32).prefetch(2)



history = model.fit(train,                                    

                    steps_per_epoch=47, 

                    epochs=50,

                    validation_data=val,

                    validation_steps=12,

                    validation_freq=1,

                    verbose=1,

                    callbacks=callbacks)
# Submission

test_pr = model.predict(test, verbose=1)

sub.loc[:, 'healthy':] = test_pr

sub.to_csv('submission.csv', index=False)

sub.head()