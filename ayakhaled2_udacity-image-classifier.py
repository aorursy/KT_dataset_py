!git clone https://github.com/tensorflow/datasets.git

%cd datasets
!pip install -e.
%cd datasets

!python -m tensorflow_datasets.scripts.download_and_prepare --register_checksums --datasets=oxford_iiit_pet
# TODO: Make all necessary imports.

import warnings

warnings.filterwarnings('ignore')

%pip --no-cache-dir install tfds-nightly
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import time

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_datasets as tfds

tfds.disable_progress_bar()
print('TensorFlow version:', tf.__version__)

print('tf.keras version:', tf.keras.__version__)

print('Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')
# TODO: Load the dataset with TensorFlow Datasets.

# https://www.tensorflow.org/datasets/api_docs/python/tfds/load

'''tfds.load(

    name, split=None, data_dir=None, batch_size=None, shuffle_files=False,

    download=True, as_supervised=False, decoders=None, read_config=None,

    with_info=False, builder_kwargs=None, download_and_prepare_kwargs=None,

    as_dataset_kwargs=None, try_gcs=False

)

'''

dataset, dataset_info =  tfds.load('oxford_flowers102', as_supervised = True, with_info = True)



# TODO: Create a training set, a validation set and a test set.

train_set = dataset["train"]

test_set = dataset["test"]

val_set = dataset["validation"]
# TODO: Get the number of examples in each set from the dataset info.

num_train_examples = dataset_info.splits['train'].num_examples

num_val_examples = dataset_info.splits['validation'].num_examples

num_test_examples = dataset_info.splits['test'].num_examples



print('There are {:,} images in the training set'.format(num_train_examples))

print('There are {:,} images in the validation set'.format(num_val_examples))

print('There are {:,} images in the test set'.format(num_test_examples))



# TODO: Get the number of classes in the dataset from the dataset info.

num_classes = dataset_info.features['label'].num_classes



print('There are {:,} number of classes'.format(num_classes))
# TODO: Print the shape and corresponding label of 3 images in the training set.

for image, label in train_set.take(3):

    image = image.numpy()

    label = label.numpy()



    print('The shape of this image is:', image.shape)

    print('The label of this image is:', label)
# TODO: Plot 1 image from the training set. Set the title 

# of the plot to the corresponding image label. 

for image, label in train_set.take(1):

    image = image.numpy()

    label = label.numpy()

    

    flower_label = label

    plt.imshow(image)

    plt.suptitle(flower_label, fontsize=20)

    plt.show()
import json

with open('/kaggle/input/josnfile/label_map.json', 'r') as f:

    class_names = json.load(f)

class_names
# TODO: Plot 1 image from the training set. Set the title 

# of the plot to the corresponding class name. 

for image, label in train_set.take(1):

    image = image.numpy()

    label = label.numpy()

    

    flower_label = class_names[str(label)]

    plt.imshow(image)

    plt.suptitle(flower_label, fontsize=20)

    plt.show()
# TODO: Create a pipeline for each set.

batch_size = 32

image_size = 224



def format_image(image, label):

    image = tf.cast(image, tf.float32)

    image = tf.image.resize(image, (image_size, image_size))

    image /= 255

    return image, label



train_batches = train_set.shuffle(num_train_examples//4).map(format_image).batch(batch_size).prefetch(1)

val_batches = val_set.map(format_image).batch(batch_size).prefetch(1)

test_batches = test_set.map(format_image).batch(batch_size).prefetch(1)
import tensorflow_hub as hub



URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"



feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size, 3))

feature_extractor.trainable = False
# TODO: Build and train your network.



model = tf.keras.Sequential([

        feature_extractor,

        tf.keras.layers.Dense(num_classes, activation='softmax')

])



model.summary()
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



EPOCHS = 20



early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)





history = model.fit(train_batches,

                        epochs=EPOCHS,

                        validation_data=val_batches,

                        callbacks=[early_stopping])
# TODO: Save your trained model as a Keras model.

t = time.time()



saved_keras_model_filepath = './model{}.h5'.format(int(t))



model.save(saved_keras_model_filepath)
# TODO: Load the Keras model



reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath

                                                  ,custom_objects={'KerasLayer':hub.KerasLayer})

reloaded_keras_model.summary()
# TODO: Create the process_image function

image_size = 224



def process_image(image):

    image = tf.cast(image, tf.float32)

    image = tf.image.resize(image, (image_size, image_size))

    image /= 255

    image = image.numpy()

    return image
from PIL import Image



image_path = '/kaggle/input/test-images/hard-leaved_pocket_orchid.jpg'

im = Image.open(image_path)

test_image = np.asarray(im)



processed_test_image = process_image(test_image)



fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)

ax1.imshow(test_image)

ax1.set_title('Original Image')

ax2.imshow(processed_test_image)

ax2.set_title('Processed Image')

plt.tight_layout()

plt.show()
# TODO: Create the predict function

def predict(image_path, model, top_k=5):

    im = Image.open(image_path)

    test_image = np.asarray(im)

    processed_test_image = process_image(test_image)

    final_img = np.expand_dims(processed_test_image, axis=0)

    preds = model.predict(final_img)

    probs = - np.partition(-preds[0], top_k)[:top_k]

    classes = np.argpartition(-preds[0], top_k)[:top_k]

    return probs, classes
# TODO: Plot the input image along with the top 5 classes

# TODO: Plot the input image along with the top 5 classes

file_names = ['cautleya_spicata.jpg','hard-leaved_pocket_orchid.jpg','orange_dahlia.jpg','wild_pansy.jpg']

top_k = 5

for filename in file_names:

    image_path = '/kaggle/input/test-images/' + filename

    image = np.asarray(Image.open(image_path)).squeeze()

    probs, classes = predict(image_path, model, top_k)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(image)

    ax1.axis('off')

    ax2.barh(np.arange(top_k), probs)

    ax2.set_aspect(0.2)

    ax2.set_yticks(np.arange(top_k))

    keys = [str(x+1) for x in list(classes)] #add 1 to the class index to match the json dict keys

    ax2.set_yticklabels([class_names.get(key) for key in keys], size='small')

    ax2.set_title('Class Probabilities')

    ax2.set_xlim(0, 1.1)

    plt.tight_layout()