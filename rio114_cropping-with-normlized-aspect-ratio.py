#Import required library

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
%matplotlib inline
# loading the table
train_df = pd.read_csv('../input/train.csv')
Ids = train_df['Id'].unique()
# Thanks Lex Toumbourou!!
INPUT_DIR = '../input'
def plot_images_for_filenames(filenames, labels, rows=4):
    imgs = [plt.imread(f'{INPUT_DIR}/train/{filename}') for filename in filenames]
    return plot_images(imgs, labels, rows)

def plot_images(imgs, labels, rows=4):
    figure = plt.figure(figsize=(13,8))
    cols = len(imgs) // rows + 1
    
    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i],cmap='gray')

rand_rows = train_df.sample(frac=1.)[:5]
imgs = list(rand_rows['Image'])
labels = list(rand_rows['Id'])

plot_images_for_filenames(imgs, labels)
PATH ='../input/train/00aa021c.jpg'
img = cv2.imread(PATH,0) 
print(img.shape) #checking image-array shape. Here, aspect ratio is 497:746, it's almost 1:1.5
plt.imshow(img)
canny_edges = cv2.Canny(img,300,300)
plt.imshow(canny_edges)
plt.plot( [0, 746], [100,100], 'w--', lw=2 )
plt.plot( [0, 0], [100,100+746/2], 'w--', lw=2 )
plt.plot( [746, 746], [100,100+746/2], 'w--', lw=2 )
plt.plot( [0, 746], [100+746/2,100+746/2], 'w--', lw=2 )
plt.imshow(canny_edges)
v = img.shape[0] # vertial pixels
h = img.shape[1] # horizontal pixels 

ver = int(h/2)
cnt = []
for i in range(canny_edges.shape[0]-ver):
    cnt.append(canny_edges[i:i+ver,:].sum()/255) # moving rectangle 

cnt_arr = np.array(cnt)
i = cnt_arr.argmax()

Img_cropped = Image.fromarray(np.uint8(img[i:i+ver,:]))
Img_cropped
def crop(PATH):
    
    h_factor = 2 #here is for 1:"2"
    
    img = cv2.imread(PATH,0)
    
    v = img.shape[0] # vertial pixels
    h = img.shape[1] # horizontal pixels 

    #find edges with Canny algorism
    canny_edges = cv2.Canny(img,300,300)
    
    if v < h/h_factor:
        fill_length = int(abs(h/h_factor-v)*0.5)#np.random.rand()) # for upper filling
    
        fill = np.zeros(fill_length* h).reshape(fill_length, h) # black rectangle for upper filling

        canny_edges = np.r_[fill,canny_edges,fill] # fill with black rectangle
        img = np.r_[fill,img,fill] # fill with black rectangle
    
    ver = int(h/h_factor)
    cnt = []
    for i in range(canny_edges.shape[0]-ver+2):
        cnt.append(canny_edges[i:i+ver,:].sum()/255)

    cnt_arr = np.array(cnt)
    i = cnt_arr.argmax()
    return Image.fromarray(np.uint8(img[i:i+ver,:]))
#handling data & images
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline

#keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Dropout, MaxPooling2D, Flatten
from keras.utils import np_utils

#keras needs normalized size, here, former (100) is width, later (50) is height of image for opencv (cv2)
SIZE = (100, 50) 
train_df = pd.read_csv('../input/train.csv')

def image_path(i):
    NAME = train_df['Image'][i]
    DIR_PATH = '../input/train/'
    return DIR_PATH + NAME

def quad_div(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    width = img_gray.shape[1]
    height = img_gray.shape[0]
    half_width = int(width / 2)
    half_height = int(height / 2)
    
    upper_left = img_gray[:half_height, :half_width]
    upper_right = img_gray[:half_height, half_width:]
    lower_left = img_gray[half_height:, :half_width]
    lower_right = img_gray[half_height:, half_width:]
    
    return cv2.resize(upper_left, SIZE), cv2.resize(upper_right, SIZE), cv2.resize(lower_left, SIZE), cv2.resize(lower_right, SIZE)
PATH = image_path(1)
img = cv2.imread(PATH)
plt.imshow(img)
upper_left, upper_right, lower_left, lower_right = quad_div(PATH)

plt.subplot(221)
plt.imshow(upper_left)
plt.axis("off")

plt.subplot(222)
plt.imshow(upper_right)
plt.axis("off")

plt.subplot(223)
plt.imshow(lower_left)
plt.axis("off")

plt.subplot(224)
plt.imshow(lower_right)
plt.axis("off")
def div_arg(path):
    imgs = []
    labels = []
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, SIZE)
    div_list = quad_div(path)
    imgs.append(img_resized)
    imgs.append(div_list[np.random.randint(3)]) # pick up only one quadrant image
    for i in ([1,0],[0,1]): # labeled as one-hot method
        labels.append(i)
        
    return np.array(imgs), np.array(labels) # out put is original (but resized) image and one quadrant image
N = 10 

PATH = image_path(N)
imgs, labels = div_arg(PATH)

for n in range(N-1):
    PATH = image_path(n)
    imgs_add, labels_add = div_arg(PATH)
    imgs = np.r_[imgs, imgs_add]
    labels = np.r_[labels, labels_add]

X = imgs.reshape([imgs.shape[0], imgs.shape[1], imgs.shape[2], 1]) #fit dimension for keras model
Y = labels
num_fil = 16
num_classes = 2

model = Sequential()
model.add(Convolution2D(num_fil, (3,3), border_mode='valid', input_shape=(SIZE[1],SIZE[0],1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(num_fil, (3,3), border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(num_fil, (3,3), border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(num_fil, (3,3), border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adadelta', loss='mse')
batch_size =32
epochs = 5

N_train = int(len(X) * 0.8)

model.fit(X[:N_train], Y[:N_train], batch_size=batch_size, epochs=epochs, validation_data=(X[N_train:], Y[N_train:]))
PATH = image_path(1)
img = cv2.imread(PATH)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
width = img_gray.shape[1]
height = img_gray.shape[0]
inc = int(min(width, height) / 2 / 11) - 1

imgs_list = []
for num in range(10):
    dum = cv2.resize(img_gray[(inc*num):(height-inc*num), (inc*num):(width-inc*num)], SIZE)
    imgs_list.append(dum)
imgs = np.array(imgs_list)

# to fit dimension for keras model
test = imgs.reshape([imgs.shape[0], imgs.shape[1], imgs.shape[2], 1])

# prediction (probability) can be used to get optimized cropping size
pred = model.predict(test)
#i =np.argmax(pred[:,0])
i = 5
print(i)
plt.imshow(imgs[i])
SIZE_model = (300, 150)

def generate_normalized_tail(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    width = img_gray.shape[1]
    height = img_gray.shape[0]
    inc = int(min(width, height) / 2 / 11) - 1

    imgs_list = []
    for num in range(10):
        dum = cv2.resize(img_gray[(inc*num):(height-inc*num), (inc*num):(width-inc*num)], SIZE)
        imgs_list.append(dum)
    imgs = np.array(imgs_list)
    
    test = imgs.reshape([imgs.shape[0], imgs.shape[1], imgs.shape[2], 1])
    pred = model.predict(test)
    
    num =np.argmax(pred[:,0])
    dum = cv2.resize(img_gray[(inc*num):(height-inc*num), (inc*num):(width-inc*num)], SIZE_model)
    return dum.reshape(1,SIZE_model[1],SIZE_model[0])