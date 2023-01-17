# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Used to make data more uniform across screen.
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Load the REQUIRED libraries
# This magic function not found
#%tensorflow_version 2.x
#from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend, models, layers, regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
# Libraries needed to build model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten,Dropout,AveragePooling2D,DepthwiseConv2D
# library to read the kaggle authentication fike
import json
from IPython.display import display # Library to help view images
from PIL import Image # Library to help view images
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Library for data augmentation

import os, shutil # Library for navigating files

# These are functions that are needed in Google Collab
#from google.colab import drive # Library to mount google drives
# Colab library to upload files to notebook
#from google.colab import files

import time

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.optimizers import Adam

import os

# display images
from IPython.display import Image

from tensorflow.keras.utils import plot_model # This will print model architecture.

from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Move files to /kaggle/working directory
#!cp -r /kaggle/input/100-bird-species/ /kaggle/working/190-bird-species
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

list_image=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        list_image.append(os.path.join(dirname, filename))
        
# View a few of the images
for i in range(14, 39):
    plt.subplot(5, 5, + 1 + i-14)
    image = mpimg.imread(list_image[i-14])
    plt.imshow(image)
    #imgplot = plt.imshow(img)
plt.show()


# We will use the MobileNet CNN that was trained on ImageNet data which gives better performance than others like Inception_V3 and VGG16
mobile = MobileNet(include_top=False,input_shape=(224,224,3),pooling='avg', weights='imagenet',alpha=1, depth_multiplier=1)


print(mobile.summary()) 
plot_model(mobile, show_layer_names=True,show_shapes=True ) 
# Specify the traning, validation, and test directories.  
base_dir='/kaggle/input/100-bird-species'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'valid')
test_dir = os.path.join(base_dir,'test')
consolidated_dir=os.path.join(base_dir,'consolidated')

# Given the number of bird species that need to be predicted can change we will make this a parameter
class_num=len(os.listdir(train_dir))
import math
def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier
train_batch=128
# Data Augmentation on training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   #horizontal_flip=True, # Flip image horizontally 
                                   samplewise_center=True,
                                   samplewise_std_normalization=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

# Since the file images are in a dirrectory we need to move them from the directory into the model.  
# Keras as a function that makes this easy. Documentaion is here: https://keras.io/preprocessing/image/

train_generator = train_datagen.flow_from_directory(
    train_dir, # The directory where the train data is located
    target_size=(224, 224), # Reshape the image to 150 by 150 pixels. This is important because it makes sure all images are the same size.
    batch_size=train_batch, # We will take images in batches of 200.
    class_mode='categorical') # The classification is multiple categories.

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    class_mode='categorical')
prediction_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    class_mode='categorical',
    shuffle=False)
train_steps_per_epoch=round_down(len(train_generator.filenames)/train_batch)
validation_batch=32
valid_steps=round_down(len(validation_generator.filenames)/validation_batch)

backend.clear_session()
# We will use the MobileNet CNN that was trained on ImageNet data which gives better performance than others like Inception_V3 and VGG16
mobile = MobileNet(include_top=False,input_shape=(224,224,3),weights='imagenet')
model_name='MobileNet'
mobile.trainable = False
mobileMod= models.Sequential()
mobileMod.add(mobile)
mobileMod.add(layers.Flatten())
mobileMod.add(layers.Dense(256, activation = 'relu',kernel_regularizer = regularizers.l1(0.00001)))
mobileMod.add(layers.Dropout(0.4))
mobileMod.add(layers.Dense(class_num, activation = 'softmax'))

# We will still use the same generators with data augmentation defined above
epoch=40
start_3 = time.perf_counter()

mobileMod.compile(optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

history= mobileMod.fit_generator(
    train_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epoch,
    validation_data=validation_generator,
    validation_steps=valid_steps,
    verbose = 2,
    callbacks = [EarlyStopping(monitor='accuracy', patience = 5, restore_best_weights=True)])
end_3 = time.perf_counter()
# Code to plot performance in difference epochs. Here I am not seeing the graphs because only epoch was run above. But google collab should give the graph
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Evaluate the performance of the model on test dataset. Low here because of the above issue of only epoch run
test_loss, test_acc = mobileMod.evaluate_generator(test_generator, steps = valid_steps)
print('test_acc for {} is {}'.format(model_name,test_acc))
print('Loaded {} feature extractor in {:.2f}sec'.format(model_name, end_3-start_3))
# We will now look at the images that were incorrectly classfied by model
# reset the test_generator before whenever you call the predict_generator. This is important, if you forget to reset the test_generator you will get outputs in a weird order or use shuffle=false

Y_pred = mobileMod.predict_generator(prediction_generator,verbose=1)
predicted_classes = np.argmax(np.round(Y_pred), axis=1)
predicted_class_indices=np.argmax(Y_pred,axis=1)

#labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
test_labels=[labels[k] for k in prediction_generator.classes]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,"labels":test_labels,"Predictions":predictions}) #
# Look at the images that have been misclassified
rslt_df = results[results['labels'] != results['Predictions']] 
#print(rslt_df)

#Add full path to the Filename
rslt_df.loc[rslt_df.index, 'Filename'] = '/kaggle/input/100-bird-species/test/' + rslt_df['Filename'].astype(str)


# To display misclassified images in pandas dataframe we will use the following function which have adopted from https://stackoverflow.com/questions/46107348/how-to-display-image-stored-in-pandas-dataframe
import glob
import random
import base64
import pandas as pd

from PIL import Image
from io import BytesIO
from IPython.display import HTML
import io

pd.set_option('display.max_colwidth', -1)


def get_thumbnail(path):
    #path = "\\\\?\\"+path # This "\\\\?\\" is used to prevent problems with long Windows paths
    path = path # This "\\\\?\\" is used to prevent problems with long Windows paths
    i = Image.open(path)    
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
#We can pass our local image path to get_thumbnail(path) with following:

rslt_df['FilenamePill'] = rslt_df.Filename.map(lambda f: get_thumbnail(f))

#view pandas dataframe with resized images by call image_formatter function in IPython.display HTML function:
HTML(rslt_df.to_html(formatters={'FilenamePill': image_formatter}, escape=False))

predicted_class_indices=np.argmax(Y_pred,axis=1)
predictions = [labels[k] for k in predicted_class_indices]
test_labels=[labels[k] for k in test_generator.classes]

plt.subplots(figsize=(20,15))
#sns.heatmap(confusion_matrix(test_generator.classes, y_pred))
sns.heatmap(confusion_matrix(test_labels, predictions))
print('Classification Report')
print(classification_report(test_generator.classes, predicted_class_indices, target_names=list(test_generator.class_indices.keys())))