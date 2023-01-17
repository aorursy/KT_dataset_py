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
import glob
import cv2
from PIL import Image
import numpy as np
image_array=[]
l=[]
for img in glob.glob("/kaggle/input/flowers-recognition/flowers/sunflower/*.jpg"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    image_array.append(np.array(size_image))
    l.append("sunflower")
for img in glob.glob("/kaggle/input/flowers-recognition/flowers/rose/*.jpg"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    image_array.append(np.array(size_image))
    l.append("rose")
for img in glob.glob("/kaggle/input/flowers-recognition/flowers/daisy/*.jpg"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    image_array.append(np.array(size_image))
    l.append("daisy")
for img in glob.glob("/kaggle/input/flowers-recognition/flowers/dandelion/*.jpg"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    image_array.append(np.array(size_image))
    l.append("dandelion")
for img in glob.glob("/kaggle/input/flowers-recognition/flowers/tulip/*.jpg"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    image_array.append(np.array(size_image))
    l.append("tulip")
    
    
len(image_array)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
label=pd.DataFrame(l)
label.head()
la=LabelEncoder()
labels=la.fit_transform(label[0])
types=np.unique(labels)
types
data=np.array(image_array)
import matplotlib.pyplot as plt
figure=plt.figure(figsize=(15,10))
ax=figure.add_subplot(121)
ax.imshow(data[0])
bx=figure.add_subplot(122)
bx.imshow(data[60])
plt.show()
np.save("Cells",data)
np.save("labels",labels)
Cells=np.load("Cells.npy")
labels=np.load("labels.npy")
s=np.arange(Cells.shape[0])
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]
num_classes=len(np.unique(labels))
len_data=len(Cells)
num_classes
x_train,x_test=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)
y_train,y_test=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
import keras
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.applications.vgg19 import VGG19
base_model = VGG19(weights='imagenet',include_top=False, input_shape=(50,50,3))
x = base_model.output
x = Flatten()(x)
x=Dense(500, activation='relu')(x)
x=Dropout(0.2)(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=1,validation_split=0.33,callbacks=[checkpoint])
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
figure=plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.legend(['Training Accuracy','Val Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.legend(['Training Loss','Val Loss'])
image = np.expand_dims(data[100], axis=0)
p=np.argmax(model.predict(image),axis=1)
plt.imshow(data[100])
a=la.inverse_transform(p)
print(a)