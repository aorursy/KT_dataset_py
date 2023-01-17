from kaggle_datasets import KaggleDatasets



import tensorflow as tf

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()
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

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Competition data access

# TPUs read data directly from Google Cloud Storage (GCS). 

# This Kaggle utility will copy the dataset to a GCS bucket

# co-located with the TPU.

GCS_DS_PATH = KaggleDatasets().get_gcs_path()

GCS_DS_PATH
# GCS_PATH = os.path.join(GCS_DS_PATH + '/tfrecords-jpeg-512x512')

GCS_PATH = GCS_DS_PATH + '/tfrecords-jpeg-512x512'

_TRAINING_FILENAMES = tf.io.gfile.glob( \

    GCS_PATH + '/train/*.tfrec')

_VALIDATION_FILENAMES = tf.io.gfile.glob( \

    GCS_PATH + '/val/*.tfrec')

_TEST_FILENAMES = tf.io.gfile.glob( \

    GCS_PATH + '/test/*.tfrec')
IMAGE_SIZE = [512, 512]

EPOCHS = 10

BATCH_SIZE = 20
def plotBatch(dataset):

    images, labels = next(

        iter(dataset.unbatch().batch(BATCH_SIZE)))

    images = images.numpy()

    labels = labels.numpy()



    cols = 5

    rows = -((-len(labels)) // cols)



    fig, axes = plt.subplots(rows, cols)

    for row in range(rows):

        for col in range(cols):

            idx = row * cols + col

            ax = axes[row, col]

            ax.imshow(images[idx])

            ax.axis("off")

            label = labels[idx]

def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image
def read_labeled_tfrecord(example):

    format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "class": tf.io.FixedLenFeature([], tf.int64),

    }



    example = tf.io.parse_single_example(

        example, format)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)



    return image, label





def read_unlabeled_tfrecord(example):

    format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "id": tf.io.FixedLenFeature([], tf.string),

    }



    example = tf.io.parse_single_example(

        example, format)

    image = decode_image(example['image'])

    id = example['id']



    return image, id
def getTrainData():

    dataset = tf.data.TFRecordDataset(

        _TRAINING_FILENAMES)

    dataset = dataset.map(read_labeled_tfrecord)

    dataset = dataset.cache()

    dataset = dataset.repeat(EPOCHS)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(1)



    return dataset
def getValidationData():

    dataset = tf.data.TFRecordDataset(

        _VALIDATION_FILENAMES)

    dataset = dataset.map(read_labeled_tfrecord)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(1)



    return dataset
def getTestData():

    dataset = tf.data.TFRecordDataset(

        _TEST_FILENAMES)

    dataset = dataset.map(read_unlabeled_tfrecord)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(1)



    return dataset
train_dataset = getTrainData()

test_dataset = getTestData()
plotBatch(train_dataset)
plotBatch(test_dataset)
def define_model(input_shape, n_classes):

    inp = tf.keras.layers.Input(shape=input_shape)

    vgg16 = tf.keras.applications.VGG16(include_top=False)

    for layer in vgg16.layers:

        layer.trainable = False

    vgg16Out = vgg16(inp)

    avgPool = tf.keras.layers.GlobalAveragePooling2D()(vgg16Out)

    dense1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(avgPool)

    out = tf.keras.layers.Dense(n_classes)(dense1)



    model = tf.keras.Model(inp, out)

    

    return model
model = define_model((*IMAGE_SIZE, 3), 104)

loss = tf.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(0.001)

accuracy = tf.metrics.Accuracy()

step = tf.Variable(1, name="global_step")
@tf.function

def train_step(features, labels):

    with tf.GradientTape() as tape:

        logits = model(features)

        loss_value = loss(labels, logits) 

    

    gradients = tape.gradient(loss_value,

                    model.trainable_variables)

    optimizer.apply_gradients(

        zip(gradients, model.trainable_variables))

    step.assign_add(1)

    accuracy_value = accuracy(labels,

                        tf.argmax(logits, -1))



    return loss_value, accuracy_value
# @tf.function

def loop(inputs):

    for features, labels in inputs:

        loss_value, accuracy_value = train_step(

            features, labels)

        if step.numpy() % 10 == 0:

            tf.print("step: {} loss: {} acc: {}".format(

                 step.numpy(), loss_value.numpy(), 

                    accuracy_value.numpy()))
loop(train_dataset)
test_dataset = getTestData()

images = test_dataset.map(lambda img, id: img)

preds = list()

for imgs in images:

    logits = model(imgs)

    logits = logits.numpy()

    preds.extend(logits)

preds = np.array(preds)

preds = np.argmax(preds, axis=-1)

print(preds)
np.unique(preds)
ids = test_dataset.map(lambda image, id: id)

ids = list(ids.unbatch())

ids = tf.convert_to_tensor(ids)

ids = ids.numpy().astype("U")

ids
submission_df = pd.DataFrame(dict(id=ids, label=preds))

submission_df.to_csv("submission.csv", index=False)