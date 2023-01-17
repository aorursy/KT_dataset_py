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
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.image import imread

test_path = "/kaggle/input/cell_images/test/"

train_path = "/kaggle/input/cell_images/train/"

os.listdir(test_path) 
os.listdir(train_path)
os.listdir(train_path+"parasitized")[0]
plt.imshow(imread(train_path + "parasitized/" + os.listdir(train_path+"parasitized")[10]))
plt.imshow(imread(train_path + "uninfected/" + os.listdir(train_path+"uninfected")[10]))
len(os.listdir(train_path+"parasitized"))
len(os.listdir(train_path+"uninfected"))
len(os.listdir(test_path+"parasitized"))
len(os.listdir(test_path+"uninfected"))
dim1 = []

dim2 = []



for img_name in os.listdir(test_path + "uninfected"):

    img = imread(test_path + "uninfected/"+ img_name)

    d1,d2,colors = img.shape

    dim1.append(d1)

    dim2.append(d2)

    
sns.jointplot(dim1,dim2)
np.mean(dim1)

           
np.mean(dim2)
#Average dimensions is 130X130

130*130*3
#data preprocess and Data augumentation



from tensorflow.keras.preprocessing.image import ImageDataGenerator

help(ImageDataGenerator)
image_gen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True, fill_mode="nearest")

para_cell = imread(train_path + "parasitized/" + os.listdir(train_path+"parasitized")[10])
para_cell
plt.imshow(image_gen.random_transform(para_cell))
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,AveragePooling2D

model = Sequential()

model.add(Conv2D(filters=16,kernel_size=(2,2),input_shape=(130,130,3),activation="relu"))





#model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=32,kernel_size=(2,2),input_shape=(130,130,3),activation="relu"))





model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64,kernel_size=(2,2),input_shape=(130,130,3),activation="relu"))

model.add(Dropout(0.2))



#model.add(MaxPool2D(pool_size=(2,2)))







model.add(Conv2D(filters=128,kernel_size=(2,2),input_shape=(130,130,3),activation="relu"))

model.add(Dropout(0.2))







model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=256,kernel_size=(2,2),input_shape=(130,130,3),activation="relu"))





model.add(AveragePooling2D(pool_size=(3,3)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(128,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(64,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(10,activation="relu"))







model.add(Dense(1,activation="sigmoid"))



model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=False)
batch_size=16

train_image_gen = image_gen.flow_from_directory(train_path,target_size=(130,130),color_mode="rgb",batch_size=batch_size,class_mode="binary")

test_image_gen = image_gen.flow_from_directory(test_path,target_size=(130,130),color_mode="rgb",batch_size=batch_size,class_mode="binary",shuffle=False)
train_image_gen.class_indices
results = model.fit_generator(train_image_gen,epochs=50,validation_data=test_image_gen,callbacks=[early_stop])
model.evaluate(test_image_gen,verbose=0)
metrics = pd.DataFrame(model.history.history)

metrics
metrics[["accuracy","val_accuracy"]].plot()