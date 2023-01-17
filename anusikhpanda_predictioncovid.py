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
!pip install opencv-python
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from tqdm import tqdm
import pandas as pd
import numpy as np
train_images = os.listdir("../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/")
print(train_images[:10])
import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
for i in range(4):
    plt.subplot(1,4,i+1)
    img = cv2.imread( "../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train" + "/" + train_images[i])
    plt.imshow(img)
    plt.tight_layout()
plt.show()
def get_label(img):
    name = img.split('-')[0]
    if name=='NORMAL2': return 0
    elif name=='IM': return 0
    else: return 1
data = []
labels = []

for img in tqdm(train_images):
    try:
        label = get_label(img)
        img_read = cv2.imread("../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train" + "/" + img)
        img_resized = cv2.resize(img_read, (224,224))
        img_array = img_to_array(img_resized)                      
        data.append(img_array)
        labels.append(label)
    except:
        None  
plt.imshow(data[55])
plt.show()
image_data = np.array(data)
image_labels = np.array(labels)
image_data.shape
index = np.arange(image_data.shape[0])
np.random.shuffle(index)
image_data = image_data[index]
image_labels = image_labels[index]
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_data, image_labels, test_size=0.2, random_state=101)
x_train.shape,y_train.shape
x_test.shape, y_test.shape
#Need To Do this because it's a multiclass problem where you are predicting the probabibility of 
#every possible class, so must provide label data in (N, m) shape, where N is the number of 
#training examples, and m is the number of possible classes 
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)
train_datagen = ImageDataGenerator(rescale=1/255.,
                                horizontal_flip = True,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                fill_mode = 'nearest',
                                zoom_range = 0.3,
                                rotation_range = 30
                             )
val_datagen = ImageDataGenerator(rescale=1/255.)
train_generator = train_datagen.flow(x_train, y_train, batch_size=64, shuffle=False)
val_generator = val_datagen.flow(x_test, y_test, batch_size=64, shuffle=False)
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout
#VGG16 model from Scratch
model= Sequential()

model.add(Conv2D(input_shape=(224,224,3), filters=64, kernel_size=(3,3), padding="Same", activation="relu"))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size=64, verbose=1, epochs=10, validation_split=0.1, shuffle=False)
#test Data Accuracy
test_acc = model.evaluate(x_test, y_test)[1]
test_acc
history = model.fit_generator(train_generator, steps_per_epoch = len(x_train)/64, epochs=10, shuffle=False)
