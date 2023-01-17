# Upgrade Packages

!pip install --upgrade pip -q

!pip install -U tensorflow==2.3 -q
# Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings, pathlib

warnings.filterwarnings("ignore")



import PIL.Image as Image



import tensorflow as tf

import matplotlib.pylab as plt

import tensorflow_hub as hub

from tensorflow.keras import layers

from tensorflow import keras

print("TensorFlow Version Used: ",tf.__version__)
#  Image Paths

path = '../input/random-image-for-testing-classification/'

rndm_flowers_path = '../input/random-image-for-testing-classification/flowers/'
# Classifier (TensorFlow Hub URL)

class_url = 'https://tfhub.dev/google/tf2-preview/inception_v3/classification/4'



labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())
# Image Size

image_shape = (299,299)



# Classifier 

classifier = tf.keras.Sequential([hub.KerasLayer(class_url, input_shape=image_shape+(3,)) ])



# Making a Prediction with Pre-trained Model

img_url = path + 'flower.jpg'

img = Image.open(img_url).resize(image_shape)

img = np.array(img)/ 255.0



result = classifier.predict(img[np.newaxis, ...])

predicted_class = np.argmax(result[0], axis=-1)

plt.imshow(img)

plt.axis('off')

predicted_class_name = imagenet_labels[predicted_class]

_ = plt.title("Prediction: " + predicted_class_name.title())
# example of loading the inception v3 model

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.inception_v3 import preprocess_input

from keras.applications.inception_v3 import decode_predictions



# load model

model = InceptionV3()



# load an image from file

img_url = path + 'flower.jpg'

img = load_img(img_url, target_size = image_shape)

# convert the image pixels to a numpy array

img = np.array(img)

# reshape data for the model

img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

# prepare the image for the Inception v3 model

img = preprocess_input(img)



# predict the probability across all output classes

pred = model.predict(img)



# convert the probabilities to class labels

label = decode_predictions(pred)



# retrieve the most likely result, e.g. highest probability

label = label[0][0]



# Show the Predicted Class

plt.imshow(Image.open(img_url).resize(image_shape))

plt.axis('off')

_ = plt.title("Prediction: " + label[1] + " ("+'{:.2%}'.format(label[2])+")")



#print('%s (%.2f%%)' % (label[1], label[2]*100))
# -- Using a Pre-trained model to classify a random flower image.



flower_name = 'tulip'   #choose from dandelion,rose,sunflower,tulip 



rndm_img_url = rndm_flowers_path + flower_name + '.jpg'

rndm_img = Image.open(rndm_img_url).resize(image_shape)

rndm_img = np.array(rndm_img)/ 255.0



result = classifier.predict(rndm_img[np.newaxis, ...])

predicted_class = np.argmax(result[0], axis=-1)

plt.imshow(rndm_img)

plt.axis('off')

predicted_class_name = imagenet_labels[predicted_class]

_ = plt.title("Prediction: " + predicted_class_name.title())
# Load Dataset

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"



img_dir = tf.keras.utils.get_file(origin=dataset_url,fname='flower_photos',untar= True)

img_dir = pathlib.Path(img_dir)



# Image Count

print("Total Images Downloaded: ", len(list(img_dir.glob('*/*.jpg'))))
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

image_data = img_gen.flow_from_directory(str(img_dir), target_size=image_shape)
for image_batch, label_batch in image_data:

    print("Image batch shape: ", image_batch.shape)

    print("Label batch shape: ", label_batch.shape)

    break
# Feature Extractor URL

fe_url = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'



# Feature Extractor Layer

fe_layer = hub.KerasLayer(fe_url, input_shape=image_shape+(3,))



feature_batch = fe_layer(image_batch)

#print(feature_batch.shape)



# Freezing the feature extractor layer,so the training only modifies the new classifier layer.

fe_layer.trainable = False



#Adding a Classification Layer

model = tf.keras.Sequential([

        fe_layer,

        layers.Dense(image_data.num_classes)])





# Model Summary

model.summary()
# Compile Model

model.compile(optimizer=tf.keras.optimizers.Adam(), 

              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),

              metrics=['acc'])



# Custom Callback for Visualization

class CollectBatchStats(tf.keras.callbacks.Callback):

    def __init__(self):

        self.batch_losses = []

        self.batch_acc = []



    def on_train_batch_end(self, batch, logs=None):

        self.batch_losses.append(logs['loss'])

        self.batch_acc.append(logs['acc'])

        self.model.reset_metrics()
# Fit / Train the Model

epoch = 10

steps = np.ceil(image_data.samples/image_data.batch_size)



callbacks = CollectBatchStats()



history = model.fit(image_data, epochs=epoch,

                   steps_per_epoch=steps,

                   callbacks=[callbacks],

                   verbose=-1)



# Plotting the Chart

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))

#fig.suptitle('Loss & Accuracy Plot')

#plt.figure()

ax1.plot(callbacks.batch_losses)

ax1.set_title("Loss vs Training Steps")

ax1.set_ylabel("Loss")

ax1.set_xlabel("Training Steps")

ax1.set_ylim([0,2])



ax2.plot(callbacks.batch_acc)

ax2.set_title("Accuracy vs Training Steps")

ax2.set_ylabel("Accuracy")

ax2.set_xlabel("Training Steps")

ax2.set_ylim([0,1])
#Random Unseen Flowers Path





flower_name = 'tulip'   #choose from dandelion,rose,sunflower,tulip 





rndm_img_url = rndm_flowers_path + flower_name + '.jpg'

rndm_img = Image.open(rndm_img_url).resize(image_shape)

rndm_img = np.array(rndm_img)/ 255.0



class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])

class_names = np.array([key.title() for key, value in class_names])



result = model.predict(rndm_img[np.newaxis, ...])

pred_id = np.argmax(result[0], axis=-1)

plt.imshow(rndm_img)

plt.axis('off')

pred_label = class_names[pred_id]



_ = plt.title("Prediction: " + pred_label.title() )
# Saving the Model

model.save('./',save_format='tf')