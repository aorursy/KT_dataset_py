# TensorFlow
import tensorflow as tf

# TensorFlow Datsets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Tensorflow Hub
import tensorflow_hub as hub

# Helper Libraries
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from os import getcwd

print('\u2022 Using TensorFlow Version:', tf.__version__)
print('\u2022 Using TensorFlow Hub:', hub.__version__)
print('\u2022 GPU Device Found.' if tf.test.is_gpu_available() else '\u2022 GPU Device Not Found. Running on CPU')
# Select the Hub/TF2 module to use
module_selection = ["mobilenet_v2", 224, 1280]
handle_base, pixels, FV_SIZE = module_selection
MODULE_HANDLE = f"https://tfhub.dev/google/tf2-preview/{handle_base}/feature_vector/4"
IMAGE_SIZE = (pixels, pixels)
print(f"Using {MODULE_HANDLE} with input size {IMAGE_SIZE} and output dimension {FV_SIZE}")
# Use tfds to load data
DATASET_NAME = 'cats_vs_dogs'
ds_builder = tfds.builder(DATASET_NAME)
ds_builder.download_and_prepare()
ds_info = ds_builder.info
# ds_builder.download_and_prepare()
(train_examples, validation_examples, test_examples) = ds_builder.as_dataset(split=["train[0%:80%]", "train[80%:90%]", "train[90%:100%]"], as_supervised=True)
num_examples = ds_info.splits['train'].num_examples
num_classes = ds_info.features['label'].num_classes

assert isinstance(train_examples, tf.data.Dataset)
print(f'Train size: {len(train_examples)}, Validation Set: {len(validation_examples)}, Test size: {len(test_examples)}')
# Resize the images to a fixed input size and rescale the input channels
def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255
    return image, label
# shuffle and batch the data
BATCH_SIZE = 32
train_batches = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)
# inspect a batch
for image_batch, label_batch in train_batches.take(1):
    pass
image_batch.shape
do_fine_tuning = False
feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                   input_shape=IMAGE_SIZE + (3,),
                                   output_shape=[FV_SIZE],
                                   trainable=do_fine_tuning)

model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.summary()
NUM_LAYERS = 10
print(f'unfreeze the top {NUM_LAYERS} layers')
if do_fine_tuning:
    feature_extractor.trainable = True
    
    for layer in model.layers[-NUM_LAYERS:]:
        layer.trainable = True
else:
    feature_extractor.trainable = False
    
print('Compile model training configuration')
if do_fine_tuning:
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.002, momentum=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
else:
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
EPOCHS = 5
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)
SAVED_MODEL = 'exp_saved_model'
# export model
tf.saved_model.save(model, SAVED_MODEL)
loaded = tf.saved_model.load(SAVED_MODEL)

print(list(loaded.signatures.keys()))
infer = loaded.signatures['serving_default']
print(infer.structured_input_signature)
print(infer.structured_outputs)
