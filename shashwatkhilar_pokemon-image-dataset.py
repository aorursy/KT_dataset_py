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
df = pd.read_csv("/kaggle/input/pokemon-images-and-types/pokemon.csv")
df.head()
import cv2                                        
import numpy as np                                
import os                                         
from random import shuffle                         
from keras.models import Sequential               
from keras.layers import Convolution2D             
from keras.layers import MaxPooling2D            
from keras.layers import Flatten                   
from keras.layers import Dense                    
from keras.layers import Dropout                   
from keras.preprocessing import image             
import matplotlib.pyplot as plt                    
import warnings#
warnings.filterwarnings('ignore')
import os
print(os.listdir("/kaggle/input"))
TRAIN_DIR = '/kaggle/input/pokemon-images-and-types/images'
IMG_SIZE = 28,28
image_names = []
data_labels = []
data_images = []
def  create_data(DIR):
     for folder in os.listdir(TRAIN_DIR):
        for file in os.listdir(os.path.join(TRAIN_DIR,folder)):
            if file.endswith("png"):
                image_names.append(os.path.join(TRAIN_DIR,folder,file))
                data_labels.append(file[0:-4])
                img = cv2.imread(os.path.join(TRAIN_DIR,folder,file))
                im = cv2.resize(img,IMG_SIZE)
                data_images.append(im)
            else:
                continue
create_data(TRAIN_DIR)
data_labels
data = np.array(data_images)
data.shape
df.Name
df["Name"].value_counts()
items=df["Name"]
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.utils import np_utils

le = LabelEncoder()
label = le.fit_transform(data_labels)
label = to_categorical(label)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=0.20,random_state=42)

print("X_train shape",X_train.shape)
print("X_test shape",X_test.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_test.shape)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(721, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(721,activation=tf.nn.softmax))
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
model.fit(x=X_train,y=y_train, epochs=10)
model.evaluate(X_test, y_test)[1]*100
items
image_index = int(input("Enter the index of image within 145 \n"))
probability=model.predict(np.array([X_test[image_index]]))
print("The below is an image of a "+data_labels[np.argmax(probability)]+" with a probability of "+str(np.amax(probability)))
plt.imshow(X_test[image_index])
print()
probability
df.tail()
print("The below is an image of a "+items[np.argmax(probability)]+" with a probability of "+str(np.amax(probability)))
data_labels
