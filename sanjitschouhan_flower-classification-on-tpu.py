import os

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets
tf.test.is_gpu_available()
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
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

dataset_base_path = GCS_DS_PATH+"/tfrecords-jpeg-224x224"

def get_records(split_name):

    filenames = tf.io.gfile.glob(dataset_base_path+"/"+ split_name+"/"+ "*.tfrec")

    return tf.data.TFRecordDataset(filenames)
train_records = get_records("train")

val_records = get_records("val")



options = tf.data.Options()

options.experimental_deterministic = False



test_records = get_records("test").with_options(options)
image_feature_description = {

    'id': tf.io.FixedLenFeature([], tf.string),

    'class': tf.io.FixedLenFeature([], tf.int64, default_value=0),

    'image': tf.io.FixedLenFeature([], tf.string),

}



def decode_image(image):

    image = tf.io.decode_image(image)

    image = tf.cast(image, tf.float32)

    image = tf.reshape(image, (image_size, image_size, 3))

    image /= 255

    return image



def parse_image_function(example_proto):

    example = tf.io.parse_single_example(example_proto, image_feature_description)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int64)

    id = example['id']

    return image, label, id

image_size = 224



training_size = sum(1 for record in train_records.map(parse_image_function))

print("Training Size:", training_size)



train_records



for image, label, id in train_records.map(parse_image_function):

    plt.imshow(image)

    plt.title(label.numpy())

    break
batch_size = 32



def label_data_map(image, label, id):

    return image, label



def unlabel_data_map(image, label, id):

    return image



training_batch = train_records.map(parse_image_function).shuffle(training_size//4).map(label_data_map).batch(batch_size).prefetch(1)

validation_batch = val_records.map(parse_image_function).map(label_data_map).batch(batch_size).prefetch(1)

test_batch = test_records.map(parse_image_function).map(unlabel_data_map).batch(batch_size).prefetch(1)
num_classes = 104



with strategy.scope():

    feature_extractor = tf.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), weights='imagenet')

    feature_extractor.trainable = False

    

    model = tf.keras.Sequential([

        feature_extractor,

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(num_classes, activation='softmax')

    ])

    

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.summary()



EPOCHS = 50



history = model.fit(training_batch, epochs=EPOCHS, validation_data=validation_batch)
model.evaluate(validation_batch)
ids = []

for img, label, id in test_records.map(parse_image_function):

    ids.append(id.numpy().decode("utf-8"))
probs = model.predict(test_batch)
preds = np.argmax(probs, axis=1)
np.savetxt('submission.csv', np.rec.fromarrays([ids, preds]), fmt=['%s', '%d'], header='id,label', delimiter=',', comments='')