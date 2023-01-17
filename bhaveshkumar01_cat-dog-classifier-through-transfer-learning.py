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
# This notebook tries to approach this dataset through transfer learning i.e, i'll use a pretrained model (MobilenetV2) and mould it 
# to make predictions on this dataset.
import os
data_dir = '../input/cat-and-dog'
os.listdir(data_dir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
# an example of a cat
cat = imread('../input/cat-and-dog/training_set/training_set/cats/cat.1.jpg')
plt.imshow(cat)
# an example of a dog
dog = imread('../input/cat-and-dog/training_set/training_set/dogs/dog.1.jpg')
plt.imshow(dog)
# to find an approximate average size of the images so as to reshape all images to this shape for model interpretation
dim1 = []
dim2 = []
for image_file in os.listdir('../input/cat-and-dog/training_set/training_set/cats'):
    if image_file[0] != '_':
        img = imread('../input/cat-and-dog/training_set/training_set/cats/'+image_file)
        d1,d2,colors = img.shape
        dim1.append(d1)
        dim2.append(d2)
height = int(np.average(dim1))
height
width = int(np.average(dim2))
width
img_shape = (height,width,3)
import tensorflow.keras as tk
# this is the base model trained on image_net dataset, it also contains the weights as intuited when trained on that dataset but we 
# will not import the top layers of the model because they are less generic, instead we'll add our own layers in their place so that 
# our model becomes ready for our dataset
base_model = tk.applications.MobileNetV2(input_shape=img_shape,
                                         include_top=False,
                                         weights='imagenet')
base_model.trainable = False # this is done so that during training the basemodel weights are not imputed again we will just train 
# the layers which we add on.
base_model.summary() # here's the basemodel's summary.
global_average_layer = tk.layers.GlobalAveragePooling2D() # to extract a feature vector from the last volume so that later 
# predictions could be made on this.
prediction_layer = tk.layers.Dense(1) # this layer makes raw predictions and output a single number per image in form of logit.
model = tk.models.Sequential([base_model,global_average_layer,prediction_layer]) # structuring our new model.
# compiling our model.
model.compile(optimizer=tk.optimizers.RMSprop(),
              loss=tk.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary() # here's is the summary of our new model.
# to perform scaling and augmentation of images.
image_gen_train = tk.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                           horizontal_flip=True,
                                                           width_shift_range=0.15,
                                                           height_shift_range=0.15,
                                                           fill_mode='nearest',
                                                           zoom_range=0.5,
                                                           rotation_range=45)
# image augmentation is generally not applied to test set so only applying scaling.
image_gen_test = tk.preprocessing.image.ImageDataGenerator(rescale=1/255)
# A `DirectoryIterator` yielding tuples of `(x, y)`
# where `x` is a NumPy array containing a batch
# of images with shape `(batch_size, *target_size, channels)`
# and `y` is a NumPy array of corresponding labels.
# here x is like our x_train and y as our y_train.
train_data_gen = image_gen_train.flow_from_directory(directory='../input/cat-and-dog/training_set/training_set/',
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=128,
                                                    target_size=img_shape[:2])
# A `DirectoryIterator` yielding tuples of `(x, y)`
# where `x` is a NumPy array containing a batch
# of images with shape `(batch_size, *target_size, channels)`
# and `y` is a NumPy array of corresponding labels.
# here x is like our x_test and y as our y_test.
test_data_gen = image_gen_test.flow_from_directory(directory='../input/cat-and-dog/test_set/test_set',
                                                  class_mode='binary',
                                                  color_mode='rgb',
                                                  batch_size=128,
                                                  target_size=img_shape[:2],
                                                  shuffle=False)
earl_stop = tk.callbacks.EarlyStopping(monitor='val_loss',patience=2) # early stopping to avoid overfitting.
history = model.fit(train_data_gen,
                   epochs=15,
                   validation_data=test_data_gen,
                   callbacks=[earl_stop])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
# our model seems to perform much better on test set as compared to train set this could be because dataset provided to us in form 
# of train and test set came from different distribution so might be possible that our test set is much easier than the train set
# but still, our model has performed pretty well. hurrey!!
