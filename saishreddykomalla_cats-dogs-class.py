import shutil
import matplotlib.pyplot as plt
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import zipfile
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import zipfile
from os import getcwd
train = f"{getcwd()}/../kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"

# os.mkdir('/kaggle/working/train')
local_train = train
z = zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", 'r')
z.extractall('/kaggle/working')
z.close()
local_train = train
z = zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", 'r')
z.extractall('/kaggle/working')
z.close()
#v 3.0 r1.0
source = "/kaggle/working/train/"
os.mkdir("/kaggle/working/training")
os.mkdir("/kaggle/working/training/cats")
os.mkdir("/kaggle/working/training/dogs")
os.mkdir("/kaggle/working/dev")
os.mkdir("/kaggle/working/dev/cats")
os.mkdir("/kaggle/working/dev/dogs")

dataset=[]
for i in os.listdir(source):
    dataset.append(i)

shuffled_set = random.sample(dataset, len(dataset))

dev_length = int(len(dataset)*0.1)
train_length = int(len(dataset) - dev_length) 
training_dirs = shuffled_set[:train_length]
dev_dirs = shuffled_set[-dev_length:]

for i in training_dirs:
    if 'cat' in i:
        shutil.copyfile(source + i, "/kaggle/working/training/cats/" + i)
    else:
        shutil.copyfile(source + i, "/kaggle/working/training/dogs/" + i)
for i in dev_dirs:
    if 'cat' in i:
        shutil.copyfile(source + i, "/kaggle/working/dev/cats/" + i)
    else:
        shutil.copyfile(source + i, "/kaggle/working/dev/dogs/" + i)
#V 3.0 r1.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), input_shape = (224,224,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
print(train_length, dev_length)
# V 3.0 r1.0
train_datagen = ImageDataGenerator(rescale = 1.0/ 255.0)
train_generator = train_datagen.flow_from_directory(
    "/kaggle/working/training",
    batch_size = 10,
    class_mode = 'binary',
    target_size = (224,224)
)
dev_datagen = ImageDataGenerator(rescale = 1.0/ 255.0)
dev_generator = dev_datagen.flow_from_directory(
    "/kaggle/working/dev",
    batch_size = 10,
    class_mode = 'binary',
    target_size = (224,224)
)
history = model.fit_generator(train_generator, epochs = 10, verbose =1, validation_data = dev_generator)
%matplotlib inline
import matplotlib.image  as mpimg
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

#______________________
#Accuracy comparision
#______________________

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc,'b', "dev Accuracy")
plt.title("Training and dev Accuracy")
plt.legend(loc=0)
plt.figure()

#_____________________________
#Loss comparision
#_____________________________

plt.plot(epochs, loss, 'r', "Training loss")
plt.plot(epochs, val_loss, 'b', "Dev loss")
plt.title("Training and dev loss")
plt.legend(loc=0)
# plt.figure()
model.save('fmodel.h5')
os.mkdir("/kaggle/working/testing")
os.mkdir("/kaggle/working/testing/test")
for i in os.listdir("/kaggle/working/test"):
    shutil.copyfile("/kaggle/working/test/" + i, "/kaggle/working/testing/test/" + i)
test_datagen = ImageDataGenerator(rescale = 1.0/ 255.0)
test_generator = test_datagen.flow_from_directory(
    "/kaggle/working/testing",
    target_size = (224,224),
    batch_size = 10,
    class_mode = 'binary'
)
preds = model.predict(test_generator)
preds[0]
# load_m = tf.keras.models.load_model('/kaggle/input/my-model/fmodel.h5')
# load_m.summary()
# pred = load_m.predict_generator(test_generator, max_queue_size = 10)
# model.save_weights('smodel_weights.h5')
# model.save('smodel.h5')
counter = range(1, 12500 + 1)
solution = pd.DataFrame({"id": counter, "label":list(preds)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv("dogsVScats.csv", index = False)