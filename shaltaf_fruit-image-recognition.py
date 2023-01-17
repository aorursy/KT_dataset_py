import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

from sklearn.utils import shuffle

from keras.models import Sequential

from keras import optimizers

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D

from keras.layers import MaxPooling2D, Dropout

from sklearn.model_selection import KFold

from keras.models import Model

from PIL import Image

import cv2

import IPython

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

print(os.listdir('../input'))
from PIL import Image

X_train=[]

X_test=[]

train=[]

test=[]

size=50

for root, dirs, files in os.walk("../input/fruits-360_dataset/fruits-360/Training"):

    for name in dirs:

        for filename in os.listdir(os.path.join(root, name)):

            image=Image.open( os.path.join(root, name) + "/"+filename)

            img_resized = np.array(image.resize((size,size)))

            X_train.append(img_resized)

            train.append(name)

            

for root, dirs, files in os.walk("../input/fruits-360_dataset/fruits-360/Test"):

    for name in dirs:

        for filename in os.listdir(os.path.join(root, name)):

            image=Image.open( os.path.join(root, name) + "/"+filename)

            img_resized = np.array(image.resize((size,size)))

            X_test.append(np.array(img_resized))

            test.append(name)

            

X_train=np.array(X_train)

X_test=np.array(X_test)
X_train,train=shuffle(X_train,train,random_state=44)

X_test,test=shuffle(X_test,test,random_state=44)
test=np.array(test)

train=np.array(train)

from sklearn.preprocessing import OneHotEncoder

hot = OneHotEncoder()

y_train=train.reshape(len(train), 1)

y_train = hot.fit_transform(y_train).toarray()

y_test=test.reshape(len(test), 1)

y_test = hot.transform(y_test).toarray()

X_train=X_train/255
size=X_train.shape[1]

cvscores = []

model = Sequential()

model.add(ZeroPadding2D(2, input_shape=(size, size, 3)))

model.add(Conv2D(32, (7, 7),strides=(1, 1),padding="valid", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),strides=(1, 1),padding="valid", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3,3),strides=(1, 1),padding="valid", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(256, (1,1),strides=(1, 1),padding="valid", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(114, activation="softmax")) 

rmsprop = optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,batch_size = 1024, epochs=30,verbose=0 )

scores = model.evaluate(X_test, y_test, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))  
for k in range(10):

    i=np.random.randint(len(test))

    plt.imshow(X_test[i,:,:,:])

    plt.show()

    cc=model.predict(X_test[i:i+1,:,:,:])

    cc=hot.inverse_transform(cc)

    print("prediction: ",cc)

    print("Reference       :",test[i])
size=X_train.shape[1]

cvscores = []

model = Sequential()

model.add(ZeroPadding2D(2, input_shape=(size, size, 3)))

model.add(Conv2D(32, (7, 7),strides=(1, 1),padding="valid", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3),strides=(1, 1),padding="valid", activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

from keras import optimizers

model.add(Dense(114, activation="softmax")) 

rmsprop = optimizers.RMSprop(lr=0.0001, decay=1e-6)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,batch_size = 1024, epochs=30,verbose=0 )
scores = model.evaluate(X_test, y_test, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))  
for k in range(10):

    i=np.random.randint(len(test))

    plt.imshow(X_test[i,:,:,:])

    plt.show()

    cc=model.predict(X_test[i:i+1,:,:,:])

    cc=hot.inverse_transform(cc)

    print("prediction: ",cc)

    print("Reference       :",test[i])