# https://victorzhou.com/blog/intro-to-cnns-part-1/

# https://imgur.com/BYoYBcH.jpg

import cv2,os

import requests

from PIL import Image

from io import BytesIO

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline # Only use this if using iPython

## Reading the  image from the internet



url  = 'https://cache-graphicslib.viator.com/graphicslib/page-images/742x525/146228_Venice_Burano_Venetian%20Lagoon_shutterstock_127690031.jpg'



response = requests.get(url)

img = np.array(Image.open(BytesIO(response.content)))

plt.figure(figsize=(9,4))

plt.imshow(img)
print("img.shape = ", img.shape)
# Plot the three channels of the image

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 20))

for i in range(0, 3):

    ax = axs[i]

    ax.imshow(img[:, :, i], cmap = 'gray') 

plt.show()
# Transform the image into HSV and HLS models

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# Plot the converted images

fig, (ax1, ax2,ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 20))

ax1.set_title("Gray Scale");ax1.imshow(img_gray, cmap= 'gray')

ax2.set_title("HSV");ax2.imshow(img_hsv, cmap= 'hsv')

ax3.set_title("HLS");ax3.imshow(img_hls,cmap= 'twilight')

plt.show()
# Reading a sample image from internet

url_luna  = 'https://imgur.com/Z0mYQTn.jpg'

response = requests.get(url_luna)

img_luna = np.array(Image.open(BytesIO(response.content)))



# converting to gray scale

gray_luna = cv2.cvtColor(img_luna, cv2.COLOR_RGB2GRAY)



# remove noise

#gray_luna = cv2.GaussianBlur(gray_luna,(5,5),0)



# convolute with proper kernels

#laplacian = cv2.Laplacian(gray_luna,cv2.CV_64F)

sobelx = cv2.Sobel(gray_luna,cv2.CV_64F,1,0,ksize=3)  # x

sobely = cv2.Sobel(gray_luna,cv2.CV_64F,0,1,ksize=3)  # y





plt.figure(figsize = (15,6))

plt.subplot(1,2,1);plt.imshow(gray_luna,cmap = 'gray')

plt.title('Original Blurred'); plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2);plt.imshow(sobely,cmap = 'gray')

plt.title('SobelY'); plt.xticks([]), plt.yticks([])

plt.show()
plt.figure(figsize = (15,6))

plt.subplot(1,3,1);plt.imshow(gray_luna,cmap = 'gray')

plt.title('Original Blurred'); plt.xticks([]), plt.yticks([])

plt.subplot(1,3,2);plt.imshow(sobely,cmap = 'gray')

plt.title('SobelY'); plt.xticks([]), plt.yticks([])

plt.subplot(1,3,3);plt.imshow(sobelx,cmap = 'gray')

plt.title('SobelX'); plt.xticks([]), plt.yticks([])



plt.show()
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()



print("Reading the MNIST dataset from keras.datasets...")

print('X_train.shape = ', X_train.shape,

     '\ny_train.shape = ', y_train.shape,

     '\nX_test.shape = ', X_test.shape,

     '\ny_test.shape = ', y_test.shape)





# Making sure that the values are float so that we can get decimal points after division

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



# Normalizing the RGB codes by dividing it to the max RGB value.

X_train /= 255

X_test /= 255



image_index = 7777 # You may select anything up to 60,000

sample_img = X_train[image_index]

sample_label =  y_train[image_index]

print("Image number ",(image_index -1)  ," is ", sample_label) # The label is 8

plt.imshow(sample_img, cmap='Greys')
class Conv3x3:

  # A Convolution layer using 3x3 filters.



    def __init__(self, num_filters, kernel):

        self.num_filters = num_filters



        # filters is a 3d array with dimensions (num_filters, 3, 3)

        # We divide by 9 to reduce the variance of our initial values

        #self.filters = np.random.randn(num_filters, 3, 3) / 9

        self.filters = kernel / 9 





    def iterate_regions(self, image):

        '''

        Generates all possible 3x3 image regions using valid padding.

        - image is a 2d numpy array

        '''

        h, w = image.shape



        for i in range(h - 2):

            for j in range(w - 2):

                im_region = image[i:(i + 3), j:(j + 3)]

                yield im_region, i, j



    

    def forward(self, input):

        '''

        Performs a forward pass of the conv layer using the given input.

        Returns a 3d numpy array with dimensions (h, w, num_filters).

        - input is a 2d numpy array

        '''

        h, w = input.shape

        output = np.zeros((h - 2, w - 2, self.num_filters))



        for im_region, i, j in self.iterate_regions(input):

            output[i, j] = np.sum(im_region * self.filters, axis=(0, 1))



        return output

kernel = np.reshape([-1, -1, -1,-1 , 8 , -1,-1 , -1, -1 ], (3,3)) 

print('Kernel =\n', kernel,  '\nShape of the Kernel = ',  kernel.shape)

conv = Conv3x3(8, kernel )
conv = Conv3x3(8, kernel )

print("Shape of Input image = ", sample_img.shape)

conv_output = conv.forward(sample_img)

print("Shape of Output after the convolution = ", conv_output.shape) # (26, 26, 8)



plt.figure(figsize=(15,8))



for i in range(1,9):

    

    plt.subplot(2,8,i)

    plt.imshow(conv_output[:,:,(i-1)], cmap='gray')

    
class MaxPool2:

  # A Max Pooling layer using a pool size of 2.



    def iterate_regions(self, image):

        '''

        Generates non-overlapping 2x2 image regions to pool over.

        - image is a 2d numpy array

        '''

        h, w, _ = image.shape

        new_h = h // 2

        new_w = w // 2



        for i in range(new_h):

            for j in range(new_w):

                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]

                yield im_region, i, j



    def forward(self, input_img):

        '''

        Performs a forward pass of the maxpool layer using the given input.

        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).

        - input is a 3d numpy array with dimensions (h, w, num_filters)

        '''

        h, w, num_filters = input_img.shape

        output = np.zeros((h // 2, w // 2, num_filters))



        for im_region, i, j in self.iterate_regions(input_img):

            output[i, j] = np.amax(im_region, axis=(0, 1))



        return output
conv = Conv3x3(8, kernel)

pool = MaxPool2()



conv_output = conv.forward(sample_img)

pool_output = pool.forward(conv_output)





print("Shape of Input Image = ",  sample_img.shape )

print("Shape of Convolution output volume = ",  conv_output.shape )

print("Shape of MaxPooling output volume = ",  pool_output.shape )

plt.figure(figsize=(15,8))



for i in range(1,9):

    plt.suptitle("Convolution and MaxPooling output")

    plt.subplot(2,8,i)

    plt.imshow(conv_output[:,:,(i-1)], cmap='gray')

    

    plt.subplot(2,8,i+8)

    plt.imshow(pool_output[:,:,(i-1)], cmap= 'gray')



plt.show()
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()



#reshape data to fit model

X_train = X_train.reshape(60000,28,28,1)

X_test = X_test.reshape(10000,28,28,1)
from keras.utils import to_categorical

#one-hot encode target column

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

y_train[0]
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten

#create model

model = Sequential()

#add model layers

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1), name= 'FirstLayer'))

model.add(Conv2D(32, kernel_size=3, activation='relu', name= 'SecondLayer'))

model.add(Flatten(name= 'ThirdLayer'))

model.add(Dense(10, activation='soft', name= 'FourthLayer'))
#compile model using accuracy to measure model performance

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#train the model

history = model.fit(X_train, y_train,validation_split = 0.3, epochs=3)
history.history.keys()
history.history.keys()

plt.figure(figsize = (15,6))

ax = plt.subplot(111)

ax.plot(history.history['val_acc'], '-', color='b',label = 'Validation' )

ax.plot(history.history['acc'],'-', color='r', label = 'Training')

plt.title('Training vs Validation Accuracy')

ax.legend()
predictions = model.predict(X_test)

print(predictions.shape)
import sklearn

from sklearn.metrics import classification_report, confusion_matrix

#print(y_test.shape, predictions.shape)

print(classification_report([np.argmax(i) for i in y_test], [np.argmax(i) for i in predictions]))
import seaborn as sns 

confusion_mtx = confusion_matrix([np.argmax(i) for i in y_test], [np.argmax(i) for i in predictions]) 

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()