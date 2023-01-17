# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
# Set your own project id here

PROJECT_ID = 'your-google-cloud-project'

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)
sub_dir_train = os.listdir('/kaggle/input/person-images/dogImages/dogImages/train')

sub_dir_val = os.listdir('/kaggle/input/person-images/dogImages/dogImages/valid')

sub_dir_test = os.listdir('/kaggle/input/person-images/dogImages/dogImages/test')

# class_id, class_name = sub_dir.split(".") 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

example1_img_path = '/kaggle/input/person-images/dogImages/dogImages/train/090.Italian_greyhound/Italian_greyhound_06138.jpg'

img=mpimg.imread(example1_img_path)

imgplot = plt.imshow(img)
data_distribution = [(str(sub_dir).split(".")[0], len(sub_dir)) for sub_dir in sub_dir_train]

data_distribution.sort(key=lambda data_distribution: data_distribution[1]) 

data_distribuiton_labels = [data[0] for data in data_distribution]

data_distribuiton_n_samples = [data[1] for data in data_distribution]



from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt

import numpy as np





fig, ax = plt.subplots()

plt.bar(data_distribuiton_labels, data_distribuiton_n_samples)

plt.show()



n_classes = len(sub_dir_train)

print("Number of breeds:", n_classes)

print("More samples per class:", max(data_distribuiton_n_samples))

print("Less samples per class:", min(data_distribuiton_n_samples))



print("\n")

print("Number of classes in training:", len(sub_dir_train))

print("Number of classes in validation:", len(sub_dir_val))

print("Number of classes in testing:", len(sub_dir_test))



def count_files_in_dataset(dataset_dir):

    dataset_sub_dirs = os.listdir(dataset_dir)

    return sum([len(os.listdir(os.path.join(dataset_dir, dataset_sub_dir))) for dataset_sub_dir in dataset_sub_dirs])



print("\n")

print("Number of files in training:", count_files_in_dataset('/kaggle/input/person-images/dogImages/dogImages/train'))

print("Number of files in validation:", count_files_in_dataset('/kaggle/input/person-images/dogImages/dogImages/valid'))

print("Number of files in testing:", count_files_in_dataset('/kaggle/input/person-images/dogImages/dogImages/test'))
import tensorflow as tf

from keras.applications.mobilenet_v2 import MobileNetV2

from keras.preprocessing import image

import numpy as np

from keras import models

from keras import layers

from keras import optimizers



def dog_breed_classifier_mobilenet_v2(image_size, n_classes=None):

    #Load the MobileNetV2 model

    mobile_net_v2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    # Freeze the layers except the last 8 layers

    for layer in mobile_net_v2.layers[:-8]:

        layer.trainable = False





    # Create the model

    model = models.Sequential()



    # Add the vgg convolutional base model

    model.add(mobile_net_v2)



#     # Add new layers

#     model.add(layers.Dropout(0.5))

#     model.add(layers.Flatten())

#     model.add(layers.Dense(1024, activation='relu'))

#     model.add(layers.Dropout(0.5))

#     model.add(layers.Dense(n_classes, activation='softmax'))

    

    # Alternative Design

    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1024, activation='relu')) # dense layer 2

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu')) # dense layer 3

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(n_classes, activation='softmax')) # final layer with softmax activation



    # Show a summary of the model. Check the number of trainable parameters

    model.summary()

    

    return model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255,

                                   zoom_range=0.2,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   horizontal_flip=True,

                                   vertical_flip=True)



validation_datagen = ImageDataGenerator(rescale=1./255)



# Change the batchsize according to your system RAM

train_batchsize = 100

val_batchsize = 10

image_size = 224



train_generator = train_datagen.flow_from_directory(

        '/kaggle/input/person-images/dogImages/dogImages/train',

        target_size=(image_size, image_size),

        batch_size=train_batchsize,

        class_mode='categorical')



validation_generator = validation_datagen.flow_from_directory(

        '/kaggle/input/person-images/dogImages/dogImages/valid',

        target_size=(image_size, image_size),

        batch_size=val_batchsize,

        class_mode='categorical',

        shuffle=False)
# Get the model

model = dog_breed_classifier_mobilenet_v2(image_size, n_classes)



# Compile the model

from keras import optimizers

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=1e-4),

              metrics=['accuracy'])



model_file_name = '/kaggle/input/current-model-for-the-dog-breed-classifier/current_model.h5'

if not os.path.exists(model_file_name):

    # Train the model

    history = model.fit_generator(

        train_generator,

        steps_per_epoch=train_generator.samples/train_generator.batch_size ,

        epochs=150,

        validation_data=validation_generator,

        validation_steps=validation_generator.samples/validation_generator.batch_size,

        verbose=1)

    

    # Save the model

    model.save(model_file_name)

    with open('/kaggle/input/current-model-for-the-dog-breed-classifier/trainHistoryDict', 'wb') as file_pi:

        pickle.dump(history.history, file_pi)

    # Show final result

    history.history['accuracy'][-1]

else:

    model.load_weights(model_file_name)
# Plot the results

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
num_of_test_samples = len(os.listdir("/kaggle/input/person-images/dogImages/dogImages/test"))

test_batch_size = 1

test_datagen = ImageDataGenerator(rescale=1./255)





test_generator = test_datagen.flow_from_directory(

    directory='/kaggle/input/person-images/dogImages/dogImages/test',

    target_size=(image_size, image_size),

    batch_size=test_batch_size,

    class_mode='categorical',

    shuffle=False

)



from sklearn.metrics import classification_report, confusion_matrix

#Confution Matrix and Classification Report

Y_pred = model.predict_generator(test_generator)

y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')

print(confusion_matrix(test_generator.classes, y_pred))

print('Classification Report')

print(classification_report(test_generator.classes, y_pred))

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import keras



label_mapping = dict(map( lambda x: (int(x.split('.')[0]), x.split('.')[1]), sub_dir_train ))



def predictor(img_path, top_results=3):

    """

    This function takes in an image and returns a prediction on dog breed.

    INPUT: the path to the image to be classified

    OUTPUT: returns dog breed

    """

    im = keras.preprocessing.image.load_img(img_path, target_size=(image_size,image_size)) # -> PIL image

    doc = keras.preprocessing.image.img_to_array(im) # -> numpy array

    doc = np.expand_dims(doc, axis=0)

    doc = doc/255.0

    display(keras.preprocessing.image.array_to_img(doc[0]))



    # make a prediction of dog_breed based on image

    prediction = model.predict(doc)[0]

    dog_breed_indexes = prediction.argsort()[-top_results:][::-1]

    probabilities = sorted(prediction, reverse=True)[:top_results]



    for i in range(top_results):

        print("This dog looks like a {} with probability {:.2f}.".format(label_mapping[dog_breed_indexes[i]], probabilities[i]))



    

    return



predictor("/kaggle/input/ermis-photos/ermis1.jpg",5)

predictor("/kaggle/input/ermis-photos/ermis2.jpg",5)

predictor("/kaggle/input/ermis-photos/ermis3.jpg",5)

predictor("/kaggle/input/ermis-photos/ermis4.jpg",5)

predictor("/kaggle/input/ermis-photos/ermis5.jpg",5)