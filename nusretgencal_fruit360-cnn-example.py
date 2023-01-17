# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



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
from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import matplotlib.pyplot as plt

from glob import glob



train_path = '../input/fruits/fruits-360/Training/'

test_path = '../input/fruits/fruits-360/Test/'

"""img = load_img(train_path + 'Apple Braeburn/0_100.jpg')

plt.imshow(img)

plt.axis("off")

plt.show()"""
x = img_to_array(img)

print(x.shape)



className = glob(train_path + '/*') # train path'deki tüm dosyaları className'in içine at
print(x.shape)

numofclass = len(className)

print("Number of Class:", numofclass)  
model = Sequential() # model oluşturuyoruz

#filters = 32, 32 tane filtreden oluşan, kernel_size = (3,3) olsun 

model.add(Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = x.shape )) # resmim 2 boyutlu olduğu için conv2d 

model.add(MaxPooling2D(pool_size=(2, 2))) # default 2x2



model.add(Conv2D(64, (3,3), padding = "same", activation = "relu" )) # resmim 2 boyutlu olduğu için conv2d 

model.add(MaxPooling2D(pool_size=(2, 2))) # default 2x2



model.add(Conv2D(128, (3,3), padding = "same", activation = "relu")) # resmim 2 boyutlu olduğu için conv2d 

model.add(MaxPooling2D(pool_size=(2, 2))) # default 2x2



model.add(Conv2D(256, (3,3), padding = "same", activation = "relu")) # resmim 2 boyutlu olduğu için conv2d 

model.add(MaxPooling2D(pool_size=(2, 2))) # default 2x2



model.add(Flatten()) # düzleştirdik

model.add(Dense(1024, activation = "relu")) # 1024 neurondan oluşan dense olsun

model.add(Dropout(0.3)) # %50 sini kapat

model.add(Dense(numofclass, activation = "softmax")) # output numofclass kadar olması gerekiyor



model.compile(loss = "categorical_crossentropy",

              optimizer = "rmsprop",

              metrics = ["accuracy"])



batch_size = 20 # her defasında 20 tane resmimi train edeceğim

## Data generation farklı imgler yaratacağız



# rescale ile rgb değerlerini normalize ediyoruz 0-1 arasında, shear range %30 oranında yana dönüyor rastgel



train_datagen = ImageDataGenerator(rescale = 1./255,

                   shear_range = 0.3,

                   horizontal_flip = True,

                   zoom_range = 0.3)  



test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(train_path,  # training datamızı değiştirerek ekleyerek data elde etttik farklı farklı

                                                    target_size = x.shape[:2],#100,100

                                                    batch_size = batch_size, #20

                                                    color_mode = "rgb",

                                                    class_mode = "categorical")



test_generator = test_datagen.flow_from_directory(test_path, # test datamızı değiştirerek ekleyerek data elde etttik farklı farklı

                                                  target_size = x.shape[:2],# 100,100

                                                  batch_size = batch_size, #20

                                                  color_mode = "rgb",

                                                  class_mode = "categorical")
steps_for_epoch = 1000 // batch_size # 1000/20 = 50 , bir epochda 80 kez verimi train edeceğim, 

epochs = 100

validation_steps = 500 // batch_size



history = model.fit_generator(generator = train_generator,

                    steps_per_epoch = steps_for_epoch,

                    epochs = epochs,

                    validation_data = test_generator,

                    validation_steps = validation_steps)
model.save_weights("evaluate.h5")
print(history.history.keys())

plt.plot(history.history["loss"], label = "Train Loss")

plt.plot(history.history["val_loss"], label = "Validation Loss")

plt.legend()

plt.show()



plt.plot(history.history["accuracy"], label = "Train acc")

plt.plot(history.history["val_accuracy"], label = "Validation acc")

plt.legend()

plt.show()