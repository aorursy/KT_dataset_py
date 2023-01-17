%config IPCompleter.greedy=True

%matplotlib inline
import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input/"))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import cv2 

from PIL import Image

from random import randint

from sklearn.utils import shuffle

from keras.utils import np_utils

from tqdm import tqdm #Progress Bars
cat_cascade_extend = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalcatface_extended.xml')
input_train = '../input/dogs-vs-cats/train/train/'

input_test  = '../input/dogs-vs-cats/test/test/'



scale_factory = 1.05

neighbor = 5

img_size = 128

num_classes = 2

img_resize = (img_size, img_size)
img_path = input_train + 'cat.2.jpg'

# Show test data

def show_image(img_path):

    img = cv2.imread(img_path)

    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(cv_rgb)

    plt.show()

    

show_image(img_path)
def resize_to_square(image, size):

    h, w, c = image.shape

    ratio = size / max(h, w)

    resized_image = cv2.resize(image, (int(w*ratio), int(h*ratio)), cv2.INTER_AREA)

    return resized_image
def padding(image, min_height, min_width):

    h, w, c = image.shape



    if h < min_height:

        h_pad_top = int((min_height - h) / 2.0)

        h_pad_bottom = min_height - h - h_pad_top

    else:

        h_pad_top = 0

        h_pad_bottom = 0



    if w < min_width:

        w_pad_left = int((min_width - w) / 2.0)

        w_pad_right = min_width - w - w_pad_left

    else:

        w_pad_left = 0

        w_pad_right = 0

        

    return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
def get_images(train_path): 

    Labels = []  # 0 for Building , 1 dog, 2 for cat

    Images = []

    Filenames = []



    for file in tqdm(os.listdir(input_train)): #Main Directory where each class label is present as folder name.            

        words = file.split(".") 

        

        if words[0] == 'cat':

            drop = False

            image = cv2.imread(input_train + file)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #cat_images = cat_cascade.detectMultiScale(gray, scale_factory, neighbor)

            cat_image_extends = cat_cascade_extend.detectMultiScale(gray, scale_factory, neighbor)

            

            for (x,y,w,h) in cat_image_extends:   

                image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

                roi_gray = gray[y:y+h, x:x+w]

                roi_color = image[y:y+h, x:x+w]

                image_resized = cv2.resize(roi_color, (img_size, img_size)) 



                Images.append(image_resized)

                Labels.append(1.0)

                Filenames.append(file)

                drop = True

                

            if drop == False:

                image = cv2.imread(input_train + file, cv2.COLOR_BGR2GRAY)

                image = resize_to_square(image, img_size)

                image_resized = padding(image, img_size, img_size)

                

                Images.append(image_resized)

                Labels.append(1.0)

                Filenames.append(file)   

                

        elif words[0] == 'dog':

            image = cv2.imread(input_train + file, cv2.COLOR_BGR2GRAY)#COLOR_BGR2RGB, IMREAD_UNCHANGED, IMREAD_GRAYSCALE

            image = resize_to_square(image, img_size)

            image = padding(image, img_size, img_size)

            

            Images.append(image)

            Labels.append(0.0)

            Filenames.append(file) 

    

    return shuffle(Images, Labels, Filenames, random_state=817328462) #Shuffle the dataset you just prepared.
train_images, train_labels, train_filenames = get_images(input_train) #Extract the training images from the folders.
x_data = np.asarray(train_images) #converting the list of images to numpy array.

y_data = np.asarray(train_labels)
y_train = y_data[:10000]

x_train = x_data[:10000]
y_train = np_utils.to_categorical(y_train, num_classes)
x_train = x_train/255. #(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) <=> x_train/255.
print(y_train.shape)

print(x_train.shape)
def get_classlabel(class_code):

    labels = {1:'cats', 0:'dogs'}

    

    return labels[class_code]



f, ax = plt.subplots(3,3) 

f.subplots_adjust(0,0,3,3)

for i in range(0,3,1):

    for j in range(0,3,1):

        rnd_number = randint(0, len(train_images))

        ax[i,j].imshow(train_images[rnd_number])

        ax[i,j].set_title(get_classlabel(train_labels[rnd_number]))

        ax[i,j].axis('off')
from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.optimizers import SGD

from keras.utils import np_utils

from keras.applications.mobilenet import MobileNet
#build model

model = Sequential()

model.add(Flatten(input_shape = (img_size,img_size, 3)))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='sigmoid'))

model.summary()
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=16, epochs=4)
history = model.fit(x_train, y_train, validation_split=0.2, batch_size=16, epochs=10)
history2 = model.fit(x_train, y_train, validation_split=0.2, batch_size=16, epochs=20)
print(history2.history.keys())

# summarize history for accuracy

plt.plot(history2.history['acc'])

plt.plot(history2.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history2.history['loss'])

plt.plot(history2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#Save model to a file

model.save('cat-dogs-model.h5')