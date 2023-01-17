

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os



print(os.listdir("../input"))



from IPython.display import Image

Image("../input/sabihaprova/oka.png")

from IPython.display import Image

Image("../input/provathebeauty/ok.png")
from IPython.display import Image

Image("../input/provathebeauty/okkk.png")
from IPython.display import Image

Image("../input/provathebeauty/o.png")
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

import cv2
#lets define a function for rading train and test images
!pip install python-resize-image
print(os.listdir("../input"))
img = cv2.imread('../input/denoising-dirty-documents/test/1.png', 0)

plt.imshow(img,cmap='gray')
from PIL import Image

from resizeimage import resizeimage



def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        if filename == "train":

            continue

        if filename == "test":

            continue

        if filename == "train_cleaned":

            continue

        img = cv2.imread(os.path.join(folder,filename))

        img = np.array(img)

        s = img.shape

        s = np.array(s)

        if  s[0] == 258:

            img1 = Image.open(os.path.join(folder,filename))

            new1 = resizeimage.resize_contain(img1, [540, 420, 3])

            new1 = np.array(new1, dtype='uint8')

            images.append(new1)

        else:

            img1 = Image.open(os.path.join(folder,filename))

            images.append(img)

    return images



train = load_images_from_folder("../input/denoising-dirty-documents/train")

test = load_images_from_folder("../input/denoising-dirty-documents/test")

train_cleaned = load_images_from_folder("../input/denoising-dirty-documents/train_cleaned")


#now convert these image list into array and then convert values in range o-1

train = np.array(train)

test = np.array(test)

train_cleaned = np.array(train_cleaned)



train = train.astype('float32') / 255

test = test.astype('float32') / 255

train_cleaned = train_cleaned.astype('float32') / 255
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding

from keras.layers import SpatialDropout1D, Conv2D, MaxPooling2D, UpSampling2D
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(420, 540, 3,))) 

model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(UpSampling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(UpSampling2D((2, 2)))

model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))



model.summary() 



model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["accuracy"])

model.fit(train, train_cleaned, epochs=100, batch_size=52, shuffle=True, validation_data=(train, train_cleaned))
pred=model.predict(test)
array=np.array(pred)
array.shape
for img in array:

    plt.show()

    plt.imshow(img,cmap='gray')