# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



from tqdm import tqdm, tqdm_notebook

import cv2

from PIL import Image



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential , Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D , BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Input

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers, applications

from keras.applications import ResNet50

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from keras.utils import to_categorical

from keras.models import load_model
num_classes = 11

batch_size = 64

img_size = 224

num_epochs = 20
data = pd.read_csv(r"/kaggle/input/course-project-flower/data.csv")

for i in range(data.shape[0]):

    p = "/kaggle/input/course-project-flower/pic/pic/" + data.iloc[i][0][22:]

    data.loc[i, "path"] = p

data.head()
data.shape
N = data.shape[0]

X = np.empty((N, img_size, img_size, 3), dtype=np.uint8)

Y = np.empty(N, dtype=np.uint8)

arr = np.arange(3796)

np.random.seed(2020)

np.random.shuffle(arr)

cnt = 0 

for i in tqdm_notebook(arr):

    p = data.iloc[i][0]

    img = cv2.imread(p)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (img_size,img_size))

    X[cnt, :, :, :] = img

    Y[cnt] = data.iloc[i][1]

    cnt+=1

    

print(N)

# Y = np.eye(11)[Y]

Y = to_categorical(Y, num_classes=num_classes)

print("X shape:",X.shape)

print("Y shape:",Y.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=2020)

print("x_train shape",X_train.shape)

print("x_test shape",X_val.shape)

print("y_train shape",Y_train.shape)

print("y_test shape",Y_val.shape)
def create_vgg16_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = applications.VGG16(weights=None, 

                                       include_top=False,

                                       input_tensor=input_tensor)

    base_model.load_weights('../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')



    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)

    x = Dense(2048, activation='relu')(x)

    x = Dropout(0.5)(x)

    final_output = Dense(n_out, activation='softmax', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
model1 = create_vgg16_model(input_shape=(img_size, img_size, 3), n_out=num_classes)

for layer in model1.layers:

    layer.trainable = False



for i in range(-6, 0):

    model1.layers[i].trainable = True

    

metric_list = ["accuracy"]

optimizer = optimizers.Adam(lr=0.001)

model1.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
train_datagen=ImageDataGenerator(

    rotation_range=30,

    horizontal_flip=True,

    vertical_flip=True,

)



train_datagen.fit(X_train)
history1 = model1.fit_generator(

    train_datagen.flow(X_train,Y_train,batch_size=batch_size), 

    steps_per_epoch=X_train.shape[0]/batch_size , 

    epochs=num_epochs,

    validation_data=(X_val,Y_val),

    validation_steps=X_val.shape[0]/batch_size

    )

plt.plot(history1.history['val_loss'], color='b', label="val loss")

plt.plot(history1.history['loss'], color='r', label="tra loss")

plt.title("Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
plt.plot(history1.history['val_accuracy'], color='b', label="val acc")

plt.plot(history1.history['accuracy'], color='r', label="tra acc")

plt.title("Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print(history1.history['val_accuracy'])

print(history1.history['accuracy'])
from sklearn.metrics import classification_report

 

Y_pred = model1.predict(X_val)

Y_pred = np.argmax(Y_pred,axis=-1)

y_val = np.argmax(Y_val, axis=1)

print(classification_report(y_val, Y_pred))
def create_vgg19_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = applications.VGG19(weights=None, 

                                       include_top=False,

                                       input_tensor=input_tensor)

    base_model.load_weights('../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')



    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)

    x = Dense(2048, activation='relu')(x)

    x = Dropout(0.5)(x)

    final_output = Dense(n_out, activation='softmax', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
model2 = create_vgg19_model(input_shape=(img_size, img_size, 3), n_out=num_classes)

for layer in model2.layers:

    layer.trainable = False



for i in range(-6, 0):

    model2.layers[i].trainable = True

    

metric_list = ["accuracy"]

optimizer = optimizers.Adam(lr=0.001)

model2.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
train_datagen=ImageDataGenerator(

    rotation_range=30,

    horizontal_flip=True,

    vertical_flip=True,

)



train_datagen.fit(X_train)
history2 = model2.fit_generator(

    train_datagen.flow(X_train,Y_train,batch_size=batch_size), 

    steps_per_epoch=X_train.shape[0]/batch_size , 

    epochs=num_epochs,

    validation_data=(X_val,Y_val),

    validation_steps=X_val.shape[0]/batch_size

    )
plt.plot(history2.history['val_loss'], color='b', label="val loss")

plt.plot(history2.history['loss'], color='r', label="tra loss")

plt.title("Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
plt.plot(history2.history['val_accuracy'], color='b', label="val acc")

plt.plot(history2.history['accuracy'], color='r', label="tra acc")

plt.title("Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print(history2.history['val_accuracy'])

print(history2.history['accuracy'])
from sklearn.metrics import classification_report

 

Y_pred = model2.predict(X_val)

Y_pred = np.argmax(Y_pred,axis=-1)

y_val = np.argmax(Y_val, axis=1)

print(classification_report(y_val, Y_pred))