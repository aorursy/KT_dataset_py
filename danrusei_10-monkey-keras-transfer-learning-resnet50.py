# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# num_classes is the number of categories your model chooses between for each prediction
num_classes = 10

#pooling: Optional pooling mode for feature extraction when include_top is False.
#         None means that the output of the model will be the 4D tensor output of the last convolutional layer.
#        'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
#        'max' means that global max pooling will be applied.

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.add(Dense(num_classes, activation='softmax'))

# The value below is either True or False.  If you choose the wrong answer, your modeling results
# won't be very good.  Recall whether the first layer should be trained/changed or not.
my_new_model.layers[0].trainable = False
train_dir = '../input/training/training/'
val_dir = '../input/validation/validation/'

labels = pd.read_csv("../input/monkey_labels.txt")
num_classes = labels['Label'].size
labels
# We are calling the compile command for some python object. 
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#Fit the model using Data Augemnetation, will improve accuracy of the model
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                              horizontal_flip = True,
                                              width_shift_range = 0.2,
                                              height_shift_range = 0.2)
            
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

image_size = 224

train_generator = data_generator_with_aug.flow_from_directory(
       directory = train_dir,
       target_size=(image_size, image_size),
       batch_size=24,
       class_mode='categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory = val_dir,
       target_size=(image_size, image_size),
       class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=1)


#from keras.applications.resnet50 import decode_predictions
#from keras.preprocessing import image
#from IPython.display import Image, display

#img_path = '../input/validation/validation/n5/n5011.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

#preds = my_new_model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#print('Predicted:', decode_predictions(preds, top=3)[0])
#display(Image(img_path))