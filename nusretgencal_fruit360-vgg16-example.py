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

        

        



import warnings

warnings.filterwarnings("ignore")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.models import Sequential

from keras.layers import Dense

from keras.applications.vgg16 import VGG16

import matplotlib.pyplot as plt

from glob import glob



train_path = "../input/fruits/fruits-360/Training/"

test_path = "../input/fruits/fruits-360/Test/"
img = load_img(train_path + 'Avocado/0_100.jpg')

plt.imshow(img)

plt.show()
x = img_to_array(img)

print(x.shape)



numberofclass = len(glob(train_path + "/*"))

print(numberofclass)
vgg = VGG16()

print(vgg.summary())

vgg_layer_list = vgg.layers

print(vgg_layer_list)
model = Sequential()



for i in range(len(vgg_layer_list)-1):

    model.add(vgg_layer_list[i]) # add vgg_layer_list's models in our model except last model

    

print(model.summary())

for layers in model.layers: # modellerim train edilmesin zaten train edilmiş weight'lerimi kullanacağım

    layers_trainable = False



model.add(Dense(numberofclass, activation = "softmax"))  # added last element of our model

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.applications.vgg16 import preprocess_input



train_data = ImageDataGenerator(rescale=1./255,   # all pixel values will be between 0 an 1

                                shear_range=0.2, 

                                zoom_range=0.2,

                                horizontal_flip=True,

                                preprocessing_function=preprocess_input).flow_from_directory(train_path, target_size = (224,224), batch_size = 32, class_mode = 'categorical')



test_data = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input).flow_from_directory(test_path, target_size = (224,224), batch_size = 32, class_mode = 'categorical')
hist = model.fit_generator(train_data,

                           steps_per_epoch=1,# bu değerin normalde training_data'nın sayısı kadar olması gerekiyor  şimdilik 50 olabilir!!!

                           epochs = 1, # 50

                           validation_data = test_data,

                           validation_steps= 1, # bu değerin validation datanın sayısı kadar olması gerkeiyor şimdilik 25

                           verbose = 2,

                           shuffle = True)



acc = max(hist.history['accuracy'])

val_acc = max(hist.history['val_accuracy'])



print ('Training Accuracy = ' + str(acc) )

print ('Validation Accuracy = ' + str(val_acc))
print(hist.history.keys())

plt.plot(hist.history["loss"], label = "training_loss")

plt.plot(hist.history["val_loss"], label = "val_loss")

plt.legend()

plt.show()

plt.figure()

plt.plot(hist.history["accuracy"], label = "training_acc")

plt.plot(hist.history["val_accuracy"], label = "val_acc")

plt.legend()

plt.show()