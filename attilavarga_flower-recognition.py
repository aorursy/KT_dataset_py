#Load libraries



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import cv2 

import random as rn

%matplotlib inline



from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D



import os

print(os.listdir("../input"))

#Load images from input directory



data = '../input/flowers/flowers/'



DIR = os.listdir(data)



image_names = []

train_labels = []

train_images = []



imgsize = 128,128



for directory in DIR:

    for file in os.listdir(os.path.join(data,directory)):

        if file.endswith("jpg"):

            image_names.append(os.path.join(data,directory,file))

            train_labels.append(directory)

            img = cv2.imread(os.path.join(data,directory,file))

            im = cv2.resize(img,imgsize)

            train_images.append(im)

        else:

            continue
#Display some random images from dataset



fig, ax = plt.subplots(4,3)

fig.set_size_inches(12,12)

for i in range(4):

    for j in range(3):

        r = rn.randint(0,len(train_images))

        ax[i,j].imshow(train_images[r])

        ax[i,j].set_title(train_labels[r])

    

plt.tight_layout()
#categorise the labels



train = np.array(train_images)

train.astype('float32') / 255.0



labels_d = pd.get_dummies(train_labels)

labels = labels_d.values.argmax(1)

img_rows, img_cols = 128,128



from keras.utils.np_utils import to_categorical

cat  = to_categorical(labels, num_classes = 5)



#create network and train the model



model = Sequential()

model.add(Conv2D(12, kernel_size=3, activation='relu', input_shape=(128,128,3)))

model.add(Conv2D(24, kernel_size=3, activation='relu'))

model.add(Flatten())

#model.add(Dense(32,activation='relu'))

model.add(Dense(5,activation='softmax'))



model.summary()
model.compile(loss="categorical_crossentropy",

            optimizer='adam',

            metrics=['accuracy'])



model.fit(train, cat,

         epochs = 4,

         batch_size = 32,

         validation_split = 0.2)
print("Accuracy: {0:.2f}%".format(model.evaluate(train,cat)[1]*100))