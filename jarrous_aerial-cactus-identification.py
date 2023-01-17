import numpy as np

import pandas as pd



from glob import glob

from tqdm import tqdm

from PIL import Image
train_data = []

test_data = []
def creat_train_data():

    for file in tqdm(sorted(glob('../input/aerial-cactus-identification/train/train/*.jpg'))):

        img = Image.open(file)

        train_data.append( np.array(img) )
def creat_test_data():

    for file in tqdm(sorted(glob('../input/aerial-cactus-identification/test/test/*.jpg'))):

        img = Image.open(file)

        test_data.append( np.array(img) )
creat_train_data()

creat_test_data()
train_data = np.array(train_data)

test_data = np.array(test_data)

print(train_data.shape)

print(test_data.shape)
train = train_data / 255.0

test = test_data / 255.0
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import keras

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam
y = pd.read_csv('../input/aerial-cactus-identification/train.csv')

y.head()
y_train = y['has_cactus']
y_train = to_categorical(y_train, num_classes = 2)
x_train, x_val, y_train, y_val = train_test_split(train, y_train, test_size = 0.2, random_state=2)
# CNN Model

model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5,5), padding ='Same', 

                 activation ='relu', input_shape = (32,32,3)))

model.add(Conv2D(filters = 64, kernel_size = (5,5), padding ='Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 128, kernel_size = (3,3), padding ='Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding ='Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 256, kernel_size = (3,3), padding ='Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 256, kernel_size = (3,3), padding ='Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(2, activation = "softmax"))



model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=30, batch_size=64, verbose = 1)
# predict results

res = model.predict(test)



# select the indix with the maximum probability

res = np.argmax(res,axis = 1)
d = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

idx = d['id']
ans = pd.Series(res,name="has_cactus")

idx = pd.Series(idx,name = "id")



submission = pd.concat([idx,ans],axis = 1)

#print(submission)
submission.to_csv("cactus.csv",index=False)