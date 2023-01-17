# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# define and move to dataset directory

datasetdir = '/kaggle/input/trashnet-dataset/dataset-resized/'

import os

os.chdir(datasetdir)
# import the needed packages

import matplotlib.pyplot as plt

import matplotlib.image as img

from tensorflow import keras

import tensorflow.keras as keras

import numpy as np

# shortcut to the ImageDataGenerator class

ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
#At first look at the Trash categories dataset

plt.subplot(1,2,1)

plt.imshow(img.imread('metal/metal151.jpg'))

plt.subplot(1,2,2)

plt.imshow(img.imread('glass/glass501.jpg'))
#let's be more specific and print some information about our images:

images = []

for i in range(5,15):

    im = img.imread('glass/glass{}.jpg'.format(i))

    images.append(im)

    print('image shape', im.shape, 'maximum color level', im.max())
from tensorflow.keras.preprocessing.image import ImageDataGenerator



batch_size = 30



def generators(shape, preprocessing): 

    '''Create the training and validation datasets for 

    a given image shape.

    '''

    imgdatagen = ImageDataGenerator(

        preprocessing_function = preprocessing,

        horizontal_flip = True, 

        validation_split = 0.2

        ,

    )



    height, width = shape



    train_dataset = imgdatagen.flow_from_directory(

        os.getcwd(),

        target_size = (height, width), 

        classes = ('cardboard','glass','metal','paper','plastic','trash'),

        batch_size = batch_size,

        subset = 'training', 

    )



    val_dataset = imgdatagen.flow_from_directory(

        os.getcwd(),

        target_size = (height, width), 

        classes = ('cardboard','glass','metal','paper','plastic','trash'),

        batch_size = batch_size,

        subset = 'validation'

    )

    return train_dataset, val_dataset

    
def plot_history(history, yrange):

    '''Plot loss and accuracy as a function of the epoch,

    for the training and validation datasets.

    '''

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    # Get number of epochs

    epochs = range(len(acc))



    # Plot training and validation accuracy per epoch

    plt.plot(epochs, acc)

    plt.plot(epochs, val_acc)

    plt.title('Training and validation accuracy')

    plt.ylim(yrange)

    

    # Plot training and validation loss per epoch

    plt.figure()



    plt.plot(epochs, loss)

    plt.plot(epochs, val_loss)

    plt.title('Training and validation loss')

    

    plt.show()
resnet50 = keras.applications.resnet50

train_dataset, val_dataset = generators((224,224), preprocessing=resnet50.preprocess_input)
conv_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in conv_model.layers:

    layer.trainable = False

x = keras.layers.Flatten()(conv_model.output)

x = keras.layers.Dense(100, activation='relu')(x)

x = keras.layers.Dense(100, activation='relu')(x)

x = keras.layers.Dense(100, activation='relu')(x)

predictions = keras.layers.Dense(6, activation='softmax')(x)

full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)

full_model.summary()
full_model.compile(loss='categorical_crossentropy',

                  optimizer=keras.optimizers.Adamax(lr=0.001),

                  metrics=['acc'])

history = full_model.fit_generator(

    train_dataset, 

    validation_data = val_dataset,

    workers=10,

    epochs=10,

)
plot_history(history, yrange=(0.5,1))
## Saving and loading a Keras model

full_model.save('/kaggle/working/resnet50.h5')
## Saving and loading a Keras model

full_model.save_weights('/kaggle/working/resnet50weights.h5')
from keras.models import load_model

model = load_model('/kaggle/working/resnet50.h5')
from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input



img_path = 'plastic/plastic34.jpg'

img = image.load_img(img_path, target_size=(224,224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)

print(model.predict(x))

plt.imshow(img)
def show_confusion_matrix(validations, predictions):



    matrix = metrics.confusion_matrix(validations, predictions)

    plt.figure(figsize=(6, 4))

    sns.heatmap(matrix,

                cmap='coolwarm',

                linecolor='white',

                linewidths=1,

                xticklabels=LABELS,

                yticklabels=LABELS,

                annot=True,

                fmt='d')

    plt.title('Confusion Matrix')

    plt.ylabel('True Label')

    plt.xlabel('Predicted Label')

    plt.show()



y_pred_test = model.predict(x_test)

# Take the class with the highest probability from the test predictions

max_y_pred_test = np.argmax(y_pred_test, axis=1)

max_y_test = np.argmax(y_test, axis=1)



show_confusion_matrix(max_y_test, max_y_pred_test)



print(classification_report(max_y_test, max_y_pred_test))
!pip install coremltools
labels = list((k) for k,v in train_dataset.class_indices.items())

labels.sort()

coreml_model = coremltools.converters.tensorflow.convert(model,

  input_names=['image'],

  output_names=['output'],

  image_input_names='image',

  class_labels=labels,

  image_scale=1/255.0

)

coreml_model.save('/kaggle/working/TrashClassifier.mlmodel')
import coremltools

scale= 1/255



output_labels = ['cardboard','glass','metal','paper','plastic','trash']

coreml_model = coremltools.converters.tensorflow.convert('/kaggle/working/resnet50.h5',

                                                   input_name_shape_dict={'input_1': (1, 224, 224, 3)},

                                                   input_names='image',

                                                   image_input_names='image',

                                                   output_names='output',

                                                   class_labels=output_labels,

                                                   image_scale=scale)



coreml_model.author = 'fati mouhsini'

coreml_model.license = 'BSD'

coreml_model.short_description = 'Model to predict trash categories'



#coreml_model.input_description['image'] = 'image of trash to predict'

#coreml_model.output_description['output'] = 'trash type'



coreml_model.save('/kaggle/working/trash-classes.mlmodel')

