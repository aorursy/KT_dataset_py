# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import shutil
from shutil import copyfile
!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train
!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test
print(os.listdir("../working/train/train"))
os.mkdir('../working/dogs-v-cats/')
os.mkdir('../working/dogs-v-cats/training/')
os.mkdir('../working/dogs-v-cats/validation/')
os.mkdir('../working/dogs-v-cats/training/cats/')
os.mkdir('../working/dogs-v-cats/training/dogs/')
os.mkdir('../working/dogs-v-cats/validation/dogs/')
os.mkdir('../working/dogs-v-cats/validation/cats/')
BASE_DIR = '../working/'
train_dir = os.path.join(BASE_DIR,'train/train/')
test_dir = os.path.join(BASE_DIR,'test/test/')
train_cats_dir = os.path.join('../working/dogs-v-cats/training/cats/')
train_dogs_dir = os.path.join('../working/dogs-v-cats/training/dogs/')
validation_cats_dir = os.path.join('../working/dogs-v-cats/validation/cats/')
validation_dogs_dir = os.path.join('../working/dogs-v-cats/validation/dogs/')
training_dir = os.path.join('../working/dogs-v-cats/training/')
validation_dir = os.path.join('../working/dogs-v-cats/validation')
from tensorflow.keras.optimizers import RMSprop
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D((2,2),(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2),(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2),(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2),(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2),(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
for fname in os.listdir(train_dir)[:int(0.9*len(os.listdir(train_dir)))]:
    if(os.path.getsize(train_dir + fname)>0):
        if(fname.split('.')[0] == 'cat'):
            copyfile(train_dir + fname,train_cats_dir + fname)
        elif(fname.split('.')[0] == 'dog'):
            copyfile(train_dir + fname,train_dogs_dir + fname)
for fname in os.listdir(train_dir)[int(0.9*len(os.listdir(train_dir))):]:
    if(os.path.getsize(train_dir + fname)>0):
        if(fname.split('.')[0] == 'cat'):
            copyfile(train_dir + fname,validation_cats_dir + fname)
        elif(fname.split('.')[0] == 'dog'):
            copyfile(train_dir + fname,validation_dogs_dir + fname)
len(os.listdir(train_dogs_dir))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255,rotation_range=40,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.1,shear_range=0.1,horizontal_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(training_dir,target_size=(150,150),batch_size=256,class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=256,class_mode='binary')
history = model.fit(train_generator,epochs=20,validation_data = validation_generator,verbose=1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
os.listdir('../working/test/test')
n_test = len(os.listdir('../working/test/test'))
tst=[]
for fname in os.listdir('../working/test/test'):
    img = np.array(Image.open('../working/test/test/'+fname).resize((150,150)))
    tst.append(img)
test = np.asarray(tst).astype(float)#.reshape(n_test,150,150,3)
print(test.shape)
predict = model.predict(test)
sample = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
sample['label'] = predict
print(sample)
sample.to_csv("submission.csv",index=False)
