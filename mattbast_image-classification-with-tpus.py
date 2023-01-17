import numpy as np

import pandas as pd 

import tensorflow as tf

import matplotlib.pyplot as plt



from kaggle_datasets import KaggleDatasets
!pip install -q efficientnet

import efficientnet.tfkeras as efn
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

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
GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')

print(GCS_DS_PATH)
IMAGE_SIZE = 512

EPOCHS = 35

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
train_data = tf.data.TFRecordDataset(

    tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + str(IMAGE_SIZE) + 'x' + str(IMAGE_SIZE) + '/train/*.tfrec'),

    num_parallel_reads = tf.data.experimental.AUTOTUNE

)
# disable order and increase speed

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False 

train_data = train_data.with_options(ignore_order)
def read_labeled_tfrecord(example):

    tfrec_format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "class": tf.io.FixedLenFeature([], tf.int64), 

    }

    

    example = tf.io.parse_single_example(example, tfrec_format)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    

    # returns a dataset of (image, label) pairs

    return image, label 





def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  

    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])

    

    return image
# logic to read a tfrecord, decode the image in the record and return as arrays

train_data = train_data.map(read_labeled_tfrecord)
def augment(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    image = tf.image.random_brightness(image, max_delta=0.5)

    image = tf.image.random_saturation(image, lower=0.2, upper=0.5)

    

    image = tf.image.random_crop(image, size=[IMAGE_SIZE, IMAGE_SIZE, 3])

    image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    

    return image, label
train_data = train_data.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.repeat()

train_data = train_data.shuffle(2048)

train_data = train_data.batch(BATCH_SIZE)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
fig, axes = plt.subplots(1, 5, figsize=(15, 5))



for images, labels in train_data.take(1):

    for i in range(5):

        axes[i].set_title('Label: {0}'.format(labels[i]))

        axes[i].imshow(images[i])
val_data = tf.data.TFRecordDataset(

    tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + str(IMAGE_SIZE) + 'x' + str(IMAGE_SIZE) + '/val/*.tfrec'),

    num_parallel_reads = tf.data.experimental.AUTOTUNE

)



val_data = val_data.with_options(ignore_order)



val_data = val_data.map(read_labeled_tfrecord, num_parallel_calls = tf.data.experimental.AUTOTUNE)

val_data = val_data.batch(BATCH_SIZE)

val_data = val_data.cache()

val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
with strategy.scope():    

    enet = efn.EfficientNetB7(

        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),

        weights='imagenet',

        include_top=False

    )

    

    enet.trainable = True

    

    model = tf.keras.Sequential([

        enet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(104, activation='softmax', dtype='float32')

    ])
model.summary()
model.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),

]
history = model.fit(

    train_data, 

    validation_data = val_data,

    steps_per_epoch = STEPS_PER_EPOCH, 

    epochs = EPOCHS,

    callbacks = callbacks,

)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))



axes[0].set_title('Loss')

axes[0].plot(history.history['loss'], label='Train')

axes[0].plot(history.history['val_loss'], label='Validation')

axes[0].legend()



axes[1].set_title('Accuracy')

axes[1].plot(history.history['sparse_categorical_accuracy'], label='Train')

axes[1].plot(history.history['val_sparse_categorical_accuracy'], label='Validation')

axes[1].legend()



plt.show()
def read_unlabeled_tfrecord(example):

    tfrec_format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "id": tf.io.FixedLenFeature([], tf.string),  

    }

    

    example = tf.io.parse_single_example(example, tfrec_format)

    image = decode_image(example['image'])

    idnum = example['id']

    

    return image, idnum
test_data = tf.data.TFRecordDataset(

    tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-' + str(IMAGE_SIZE) + 'x' + str(IMAGE_SIZE) + '/test/*.tfrec'),

    num_parallel_reads = tf.data.experimental.AUTOTUNE

)



test_data = test_data.with_options(tf.data.Options())

test_data = test_data.map(read_unlabeled_tfrecord, num_parallel_calls = tf.data.experimental.AUTOTUNE)

test_data = test_data.batch(BATCH_SIZE)

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)
test_images = test_data.map(lambda image, idnum: image)



probabilities = model.predict(test_images)

predictions = np.argmax(probabilities, axis=-1)
ids = []



for image, image_ids in test_data.take(NUM_TEST_IMAGES):

    ids.append(image_ids.numpy())



ids = np.concatenate(ids, axis=None).astype(str)
submission = pd.DataFrame(data={'id': ids, 'label': predictions})

submission.to_csv('submission.csv', index=False)
model.save('model.h5')