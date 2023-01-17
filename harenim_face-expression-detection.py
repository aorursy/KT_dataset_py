# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
   

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
labels = []
for i in os.listdir('../input/fer2013/train/angry'):
    labels.append(0)
for i in os.listdir('../input/fer2013/train/disgust'):
    labels.append(1)
for i in os.listdir('../input/fer2013/train/fear'):
    labels.append(2)
for i in os.listdir('../input/fer2013/train/happy'):
    labels.append(3)
for i in os.listdir('../input/fer2013/train/neutral'):
    labels.append(4)
for i in os.listdir('../input/fer2013/train/sad'):
    labels.append(5)
for i in os.listdir('../input/fer2013/train/surprise'):
    labels.append(6)
import cv2
loc1 = '../input/fer2013/train/angry'
loc2 = '../input/fer2013/train/disgust'
loc3 = '../input/fer2013/train/fear'
loc4 = '../input/fer2013/train/happy'
loc5 = '../input/fer2013/train/neutral'
loc6 = '../input/fer2013/train/sad'
loc7 = '../input/fer2013/train/surprise'
features = []
from tqdm import tqdm
for i in tqdm(os.listdir(loc1)):
    f1 = cv2.imread(os.path.join(loc1,i))
    f1 = cv2.resize(f1,(100,100))
    features.append(f1)
    
for i in tqdm(os.listdir(loc2)):
    f2 = cv2.imread(os.path.join(loc2,i))
    f2 = cv2.resize(f2,(100,100))
    features.append(f2)

for i in tqdm(os.listdir(loc3)):
    f3 = cv2.imread(os.path.join(loc3,i))
    f3 = cv2.resize(f3,(100,100))
    features.append(f3)

for i in tqdm(os.listdir(loc4)):
    f4 = cv2.imread(os.path.join(loc4,i))
    f4 = cv2.resize(f4,(100,100))
    features.append(f4)
    
for i in tqdm(os.listdir(loc5)):
    f5 = cv2.imread(os.path.join(loc5,i))
    f5 = cv2.resize(f5,(100,100))
    features.append(f5)
    
for i in tqdm(os.listdir(loc6)):
    f6 = cv2.imread(os.path.join(loc6,i))
    f6 = cv2.resize(f6,(100,100))
    features.append(f6)
    
for i in tqdm(os.listdir(loc7)):
    f7 = cv2.imread(os.path.join(loc7,i))
    f7 = cv2.resize(f7,(100,100))
    features.append(f7)
    

import numpy as np
import matplotlib.pyplot as plt

import os


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image

import tensorflow as tf

img_size = 100
batch_size = 64


datagen_train = ImageDataGenerator(rescale=1./255) 
datagen_validation = ImageDataGenerator(rescale=1./255)

datagen_train = ImageDataGenerator(horizontal_flip=True)

train_generator = datagen_train.flow_from_directory("../input/fer2013/train/",
                                                    target_size=(img_size,img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory("../input/fer2013/test/",
                                                    target_size=(img_size,img_size),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(100, 100,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

opt = Adam(lr=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
%%time

epochs = 20
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size


history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    #callbacks=callbacks
)
import matplotlib.pyplot as plt

def plot_accuracy_and_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training acc',color='green')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'b', label='Training loss',color='green')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plot_accuracy_and_loss(history)
test_generator = datagen_validation.flow_from_directory("../input/fer2013/test/",
                                                    target_size=(img_size,img_size),
                                                    
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=validation_steps)
print('test acc:', test_acc)
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt

#filename="../input/dogss-test/images.jpg"
def predict_images(filename):
    img = load_img(filename, target_size=(100, 100))
    plt.imshow(img)
    plt.show()
    img = img_to_array(img)

    img = img.reshape(1, 100, 100, 3)

    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]

    result = model.predict(img)
    return result

#filename="../input/fer2013/test/happy/PrivateTest_10513598.jpg"
def answer(filename):
    result=predict_images(filename)
    if(np.argmax(result)==0):
        print("angry")
    elif(np.argmax(result)==1):
        print("disgust")
    elif(np.argmax(result)==2):
         print("fear")
    elif(np.argmax(result)==3):
         print("happy")
    elif(np.argmax(result)==4):
         print("neutral")
    elif(np.argmax(result)==5):
         print("sad")
    elif(np.argmax(result)==6):
         print("surprise")

    print(np.argmax(result))
model.save('facemodel.h5')
from keras.models import load_model
model = load_model('./facemodel.h5')
p1="../input/fer2013/test/neutral/PrivateTest_12091739.jpg"
answer(p1)
p2="../input/fer2013/test/surprise/PrivateTest_12400594.jpg"
answer(p2)
p3="../input/fer2013/test/angry/PrivateTest_12000629.jpg"
answer(p3)
p4="../input/fer2013/test/disgust/PrivateTest_29901781.jpg"
answer(p4)

p5="../input/fer2013/test/fear/PrivateTest_134207.jpg"
answer(p5)
p6="../input/fer2013/test/happy/PrivateTest_13248909.jpg"
answer(p6)
p7="../input/fer2013/test/sad/PrivateTest_13202678.jpg"
answer(p7)
import numpy as np
Y = np.array(labels)
X = np.array(features)
from keras.utils import np_utils
Xt = (X - X.mean())/X.std()        #Normalised the data
Yt = np_utils.to_categorical(Y)    #Categorical representation
Xt = Xt.reshape(8000,30000)
from keras.utils import np_utils
Xt = (X - X.mean())/X.std()        #Normalised the data
Yt = np_utils.to_categorical(Y)    #Categorical representation
Xt = Xt.reshape(8000,30000)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Xt,Yt, test_size = 0.2, random_state = 6)
from sklearn.ensemble import RandomForestClassifier
rmodel = RandomForestClassifier()
rmodel.fit(x_train,y_train)
print(rmodel.score(x_train,y_train))
print(rmodel.score(x_test,y_test))
import matplotlib.pyplot as plt
plt.imshow(x_test[70].reshape(100,100,3))
plt.show()
result = rmodel.predict(x_test[70].reshape(1,30000))
if(np.argmax(result)==0):
    print("angry")
elif(np.argmax(result)==1):
    print("disgust")
elif(np.argmax(result)==2):
    print("fear")
elif(np.argmax(result)==3):
    print("happy")
elif(np.argmax(result)==4):
    print("neutral")
elif(np.argmax(result)==5):
    print("sad")
elif(np.argmax(result)==6):
    print("surprise")