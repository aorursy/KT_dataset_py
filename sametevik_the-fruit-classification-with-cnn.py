# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Keras 

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



import matplotlib.pyplot as plt #Visualization

from glob import glob 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = "../input/fruits/fruits-360/Training/"

test_path = "../input/fruits/fruits-360/Test/"
img = load_img(train_path + "Apple Braeburn/0_100.jpg")

img
img = load_img(test_path + "Apple Braeburn/3_100.jpg")

img
plt.imshow(img);
img_to_array(img)
img_to_array(img).shape
className = glob(train_path + "/*")

className[:10]
numberOfclass = len(className)

print("There are {} different fruit files...".format(numberOfclass))
model = Sequential()



model.add(Conv2D(32, (3,3), input_shape = (100,100,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D())





model.add(Conv2D(32, (3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D())



model.add(Conv2D(64, (3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D())



model.add(Conv2D(64, (3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D())



model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(Dropout(0.50))

model.add(Dense(numberOfclass))

model.add(Activation("softmax"))
#Compiling

model.compile(loss = "categorical_crossentropy",

             optimizer = "rmsprop",

             metrics = ["accuracy"])
#Batch Size

batch_size = 32
train_datagen = ImageDataGenerator(rescale= 1./255,

                                   shear_range = 0.3,

                                   horizontal_flip= True,

                                   zoom_range= 0.3)



test_datagen = ImageDataGenerator(rescale = 1./255)





train_generator = train_datagen.flow_from_directory(train_path, 

                                                    target_size = (100,100),

                                                    batch_size = batch_size,

                                                    color_mode = "rgb",

                                                    class_mode = "categorical")





test_generator = train_datagen.flow_from_directory(test_path, 

                                                    target_size = (100,100),

                                                    batch_size = batch_size,

                                                    color_mode = "rgb",

                                                    class_mode = "categorical")





hist = model.fit_generator(

                    generator=train_generator,

                    steps_per_epoch=1600 // batch_size,

                    epochs = 50,

                    validation_data=test_generator,

                    validation_steps= 800 // batch_size,

                    )

hist.history.keys()
plt.plot(hist.history["loss"], label = "Train Loss")

plt.plot(hist.history["val_loss"], label = "Validation Loss")

plt.legend()
plt.plot(hist.history["accuracy"], label = "Train acc")

plt.plot(hist.history["val_accuracy"], label = "Validation acc")

plt.legend()