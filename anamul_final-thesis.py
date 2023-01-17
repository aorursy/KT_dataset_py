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
!ls
import cv2

import os 
def load__partial_images_from_folder(folder,target):
    images = []  
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append([img,target])
    return images
blb = load__partial_images_from_folder('../input/rice-leaf/Bacterial leaf blight/',0)
bs = load__partial_images_from_folder('../input/rice-leaf/Brown spot/',1)
ls = load__partial_images_from_folder('../input/rice-leaf/Leaf smut/',2)
blb.extend(bs)
blb.extend(ls)
data = blb
data
training_data = []
target = []
for x,y in data:
    training_data.append(x)
    target.append(y)
    
X=[]
IMG_SIZE= 32
for x in training_data:
    new_array = cv2.resize(x,(IMG_SIZE,IMG_SIZE))
    X.append(new_array)
X=[]
IMG_SIZE= 32
for x in training_data:
    new_array = cv2.resize(x,(IMG_SIZE,IMG_SIZE))
    X.append(new_array)
Xx = []
for x in X:
    tmp = x/255
    Xx.append(tmp)
Xx
import numpy as np
print(np.array(Xx).shape)

target[0:40]
label = ["Bacterial leaf blight","Brown spot","Leaf smut"]
label_0= []
label_1= []
label_2= []
for i in range(0,40):
    label_0.append(label[0])
for i in range(40,80):
    label_1.append(label[1])
for i in range(80,120):
    label_2.append(label[2])
print(label_0[39])
print(label_1[37])
print(label_2[30])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Xx,target)
import sys
from matplotlib import pyplot
#from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(3, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return model
model =  define_model()
history = model.fit(np.array(x_train),np.array(y_train),epochs=50)
from keras.utils.vis_utils import plot_model
plot_model(model)
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predicted = model.predict(np.array(x_test))
predicted
import numpy as np
result = []
for item in predicted:
    result.append(np.argmax(item))
print(result)


print(result)
import numpy as np
result = []
leaf=[]
for item in predicted:
    value =np.argmax(item)
    result.append(value)
    if value == 0:
        print((label[0]))
    elif value == 1:
        print(label[1])
    else:
        print(label[2])
       
    
    
print(result)

from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(result,y_test))