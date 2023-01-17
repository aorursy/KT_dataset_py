from keras.datasets import cifar10

from keras.layers import Input, Dense,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization

from keras.models import Model,Sequential

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import cv2

from os import listdir



from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split
import os



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
default_image_size = tuple((480, 480))

image_size = 0

directory_root = '../input/aliabhatt/'

width=480

height=480

depth=3
def convert_image_to_array(image_dir):

    try:

        image = cv2.imread(image_dir)

        if image is not None :

            image = cv2.resize(image, default_image_size)   

            return img_to_array(image)

        else :

            return np.array([])

    except Exception as e:

        print(f"Error : {e}")

        return None

image_list = []

try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root)

    

    for image_file in root_dir:

        print(f"[INFO] Processing {image_file} ...")

        image_file_image = listdir(f"{directory_root}/{image_file}/")

        

        for images_item in image_file_image:

            image_file_imag = f"{directory_root}/{image_file}/{images_item}"

            if image_file_imag.endswith(".jpg") == True or image_file_imag.endswith(".JPG") == True:

                    image_list.append(convert_image_to_array(image_file_imag))  

                    print(f"[INFO] {image_file_imag}")  

                

except Exception as e:

    print(f"Error : {e}")
print(len(image_list))
np_image_list = np.array(image_list, dtype=np.float32) /255
print("[INFO] Spliting data to train, test")

X_train, X_test = train_test_split(np_image_list, test_size=0.2, random_state = 42) 
noise_factor = 0.5

x_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

x_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)

x_test_noisy = np.clip(x_test_noisy, 0., 1.)
X_train = X_train.reshape(len(x_train_noisy),x_train_noisy.shape[1],x_train_noisy.shape[2],3)

X_test = X_test.reshape(len(x_test_noisy), x_test_noisy.shape[1],x_test_noisy.shape[2],3)

print(X_train.shape)

print(X_test.shape)
input_img = Input(shape=(480,480,3))



#Encoder

x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2,2), padding='same')(x)



x = Conv2D(8,(3,3), activation='relu', padding='same')(x)

x = MaxPooling2D((2,2), padding='same')(x)



x = Conv2D(8,(3,3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)



#Decoder

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)



x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)



x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)

x = UpSampling2D((2, 2))(x)



decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)



autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='mse')
model = Sequential()



model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(480, 480, 3)))

model.add(BatchNormalization())     # 32x32x32

model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32

model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32

model.add(BatchNormalization())     # 16x16x32

model.add(UpSampling2D())

model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32

model.add(BatchNormalization())

model.add(Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3



model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')

model.summary()
model.fit(x_train_noisy, X_train, epochs=20, batch_size=8,

            shuffle=True, validation_data=(x_test_noisy, X_test))
#encoded_imgs = model.predict(X_test)

predicted = model.predict(X_test)
plt.figure(figsize=(80,10))

for i in range(4):

    # display original images

    ax = plt.subplot(3, 20, i + 1)

    plt.imshow(X_test[i].reshape(480, 480,3))

    #plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

    # display reconstructed images

    ax = plt.subplot(3, 20, 2*20 +i+ 1)

    plt.imshow(predicted[i].reshape(480, 480,3))

    #plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

  

    

plt.show()
def single_test(image_data):

    image_array = convert_image_to_array(image_data)/255.0

    x_enpanded = image_array.reshape(1, X_test.shape[1],X_test.shape[2],3)

    y_pred = model.predict(x_enpanded)

    return y_pred,image_array
image_url = "../input/testblack/testbalck/katrina.jpg"

pred,realimage = single_test(image_url)
print(check_output(["ls", "../input/testblack/testbalck/"]).decode("utf8"))
plt.figure(figsize=(80,10))



# display original images

ax = plt.subplot(3, 20, 1)

plt.imshow(realimage.reshape(480, 480,3))

#plt.gray()

ax.get_xaxis().set_visible(False)

ax.get_yaxis().set_visible(False)

    

# display reconstructed images

ax = plt.subplot(3, 20, 41)

plt.imshow(pred.reshape(480, 480,3))

#plt.gray()

ax.get_xaxis().set_visible(False)

ax.get_yaxis().set_visible(False)

    

plt.show()