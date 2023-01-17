# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import cv2
import tensorflow as tf

import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file = os.listdir(dirname)[23]
file = os.path.join(dirname, file)
print(file)
img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap="gray")
plt.show()
img = cv2.resize(img,(100,100))
plt.imshow(img,cmap="gray")
plt.show()
datatrain = []

dirname="../input/cat-and-dog/training_set/training_set/dogs/"

for file in os.listdir(dirname):
    try:
        file = os.path.join(dirname, file)
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(100,100))

        datatrain.append([img,1])
    except:
        pass
    
dirname="../input/cat-and-dog/training_set/training_set/cats/"
for file in os.listdir(dirname):
    try:
        file = os.path.join(dirname, file)
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(100,100))

        datatrain.append([img,0])
    except:
        pass

len(datatrain)
#shguffling our data
import random
random.shuffle(datatrain)
for img in datatrain[-200:]:
    print(img[1])
#for model feeding 
X = []
y = []

for data in datatrain:
    X.append(data[0])
    y.append(data[1])
    
X =np.array(X).reshape(-1,100,100,1)
y =np.array(y)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
X = X/255.0

model = Sequential()

model.add(Conv2D((64),(2,2),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))

model.add(Conv2D((64),(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation("relu"))
model.add(tf.keras.layers.Dropout(0.5))

model.add(Dense(1))
model.add(
    Activation('sigmoid')
)

model.summary()
model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)
hist = model.fit(X,y,batch_size=32,epochs=30,validation_split=0.1)
import matplotlib.pyplot as plt

plt.subplots(figsize=(10,10))
plt.plot(hist.history['loss'],color="red",label="Loss")
plt.plot(hist.history['accuracy'],color="green",label="Accuracy",linewidth=1.5)
plt.plot(hist.history['val_loss'],color="purple",label="Val Loss")
plt.plot(hist.history['val_accuracy'],color="blue",label="Val Accuracy")
plt.legend()
plt.show()
datatest= []

dirname="../input/cat-and-dog/test_set/test_set/dogs/"

for file in os.listdir(dirname):
    try:
        file = os.path.join(dirname, file)
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(100,100))

        datatest.append([img,1])
    except:
        pass
    
dirname="../input/cat-and-dog/test_set/test_set/cats/"
for file in os.listdir(dirname):
    try:
        file = os.path.join(dirname, file)
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(100,100))

        datatest.append([img,0])
    except:
        pass

#for model testing 
X = []
y = []

for data in datatrain:
    X.append(data[0])
    y.append(data[1])
    
Xtest =np.array(X).reshape(-1,100,100,1)
ytest =np.array(y)
loss,acc = model.evaluate(Xtest,ytest)
print("Model Loss = ",loss)
print("Model Acc = ",acc)
model.save("mymodel.h5")
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
def output_func(img):
    
    model = tf.keras.models.load_model("./mymodel.h5")
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(6,6))
    plt.imshow(img,cmap="gray")
    plt.show()
    img = cv2.resize(img,(100,100))
    X   = np.array(img).reshape(-1,100,100,1)
    X   = X/255.0
    ans = model.predict(X)
    return np.round(ans)

ans = output_func('../input/photoofdog/dog.jpg')
result = {
    0:"There is a cat in the picture",
    1:"There is a dog in the picture"
}
print(result[int(ans)])

