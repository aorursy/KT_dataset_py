import numpy as np

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import array_to_img

from keras.preprocessing.image import ImageDataGenerator





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input/plant-seedlings-classification/train'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



from keras.preprocessing.image import ImageDataGenerator



dataGen = ImageDataGenerator(validation_split=0.2)



print('Starting')

path =  '/kaggle/input/plant-seedlings-classification/train'



train_it = dataGen.flow_from_directory(path, target_size=(64, 64), batch_size=32, class_mode='categorical', subset='training')

validate_it = dataGen.flow_from_directory(path, target_size=(64, 64), batch_size=32,  class_mode='categorical', subset='validation')        
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications import ResNet50

#from keras.applications.vgg19 import VGG19



resnet50_weights_path = '../input/resnet50-weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

   
model = Sequential()



model.add(ResNet50(include_top = False, pooling = 'avg', weights = resnet50_weights_path))



model.add(Dense(12, activation = 'sigmoid'))



model.layers[0].trainable = False



model.summary()
from keras import optimizers



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
fit_history = model.fit_generator(

        train_it,

        steps_per_epoch=120,

        epochs=10,

        validation_data=validate_it, validation_steps=30)
print(fit_history.history.keys())
import pandas as pd

import numpy as np

import csv



#sample_sub_path = '/kaggle/input/plant-seedlings-classification/test'



#test_data_gen = ImageDataGenerator()

#test_it = dataGen.flow_from_directory(sample_sub_path, target_size=(64, 64), class_mode='categorical')



print('Starting prediction')



#predictions = classifier.predict_classes(test_it)

labels = ['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']



predictions_list = list()



for dirname, _, filenames in os.walk('/kaggle/input/plant-seedlings-classification/test'):

    for filename in filenames:

        image = load_img(os.path.join(dirname, filename), target_size=(64,64))

        image = np.expand_dims(image, axis=0)

        prediction = model.predict_classes(image)

        predictions_list.append((filename, labels[prediction[0]]))



predictions = pd.DataFrame(predictions_list, columns=['file', 'species'])



print(predictions.head())



predictions.to_csv('submission.csv', index = False)

        