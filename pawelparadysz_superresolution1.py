from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import pandas as pd

from tqdm import tqdm, trange

import os

import warnings

warnings.filterwarnings("ignore")

os.listdir("../input")



#Version 5 replaces mean squared error with mean absolute error.
def toArray(k):

    return np.array(list(k.getdata())).reshape(k.size[1], k.size[0], 3)
from sklearn.metrics import mean_absolute_error as mae #Because we won't be able to 

from skimage.measure import compare_psnr as psnr



os.listdir("../input/images")

train_data = []

for img_path in os.listdir("../input/images"):

    train_data += [Image.open('../input/images/'+img_path)]

for img_path in os.listdir("../input/general100"):

    train_data += [Image.open('../input/general100/'+img_path)]

for img_path in os.listdir("../input/intel-data-scene/scene_classification/scene_classification/train")[:3000]:

    train_data += [Image.open('../input/intel-data-scene/scene_classification/scene_classification/train/'+img_path)]

img=train_data[30]

print(img.size)

plot5= plt.imshow(img)
x, y = img.size

img5=img.resize((100,100), resample = Image.LANCZOS)

imgp = img5.resize((x,y), resample = Image.LANCZOS)

img5=img.resize((200,200), resample = Image.LANCZOS)

imgq = img5.resize((x,y), resample = Image.LANCZOS)

print(psnr(toArray(img), toArray(imgp)), psnr(toArray(img), toArray(imgq)))

print(mae(toArray(img).reshape(x*y*3), toArray(imgp).reshape(x*y*3)), mae(toArray(img).reshape(x*y*3), toArray(imgq).reshape(x*y*3)))

try:

    print(psnr(toArray(img), toArray(img)))

except:

    print("PSNR is not continuous so I'll train with MAE")

comp = plt.figure(figsize=(9, 13))

first = comp.add_subplot(3,1,1)

first.imshow(img)

second =comp.add_subplot(3,1,2)

second.imshow(imgp)

second =comp.add_subplot(3,1,3)

second.imshow(imgq)

comp.show()
x, y = img.size

img5=img.resize((200,200), resample = Image.LANCZOS)

imgp = img5.resize((x,y), resample = Image.BICUBIC)

img5=img.resize((200,200), resample = Image.LANCZOS)

imgq = img5.resize((x,y), resample = Image.BILINEAR)

img5=img.resize((200,200), resample = Image.LANCZOS)

imgr = img5.resize((x,y), resample = Image.LANCZOS)

print(mae(toArray(img).reshape(x*y*3), toArray(imgp).reshape(x*y*3)), mae(toArray(img).reshape(x*y*3), toArray(imgq).reshape(x*y*3)),mae(toArray(img).reshape(x*y*3), toArray(imgr).reshape(x*y*3)))
def imageListToNiceSamples(images, downscale_factor = 2, img_size = 40, n_convolutions = 4): 

    X = []

    Y = []

    for image in tqdm(images):

        cutoff = n_convolutions+1

        size = np.array(image.size)

        samples_from_image = size//img_size

        newimage = image.resize(size//downscale_factor, resample = Image.LANCZOS).resize(size, resample = Image.LANCZOS)

        try:

            image_array = toArray(image)

            newimage_array = toArray(newimage)

        except:

            continue

        X_temp = []

        Y_temp = []

      #  print(size, image.size, samples_from_image)

        for j in range(samples_from_image[0]):

            for i in range(samples_from_image[1]):

                x = newimage_array[i*img_size:(i+1)*img_size,j*img_size:(j+1)*img_size,:]/130-0.99

                y = image_array[i*img_size+cutoff:(i+1)*img_size-cutoff,j*img_size+cutoff:(j+1)*img_size-cutoff,:]/130-0.99 #these fit for tanh

                x = newimage_array[i*img_size:(i+1)*img_size,j*img_size:(j+1)*img_size,:]/255+0.005

                y = image_array[i*img_size+cutoff:(i+1)*img_size-cutoff,j*img_size+cutoff:(j+1)*img_size-cutoff,:]/255+0.005 #these are for sigmoid or no activation - I've found someone not using it.

                

                X_temp+=[x.reshape(1,img_size,img_size,3)]

                Y_temp+=[y.reshape(1,img_size-2*cutoff,img_size-2*cutoff,3)]

        X+=[np.concatenate(X_temp, axis=0)] # these may look redundant, but they actually keep memory usage from blowing up and kernel from dying

        Y+=[np.concatenate(Y_temp, axis=0)]

    return(np.concatenate(X, axis=0), np.concatenate(Y, axis=0))
image_size = 30

n_convolutions = 4

X_train, y_train = imageListToNiceSamples(train_data, img_size = image_size, downscale_factor = 4)
from keras.models import Sequential, Model

from keras.layers import Conv2D, Dense, Activation, Dropout, Lambda, MaxPooling2D, BatchNormalization, Reshape, Flatten, Input

from keras.optimizers import Nadam

from keras.callbacks import EarlyStopping
def getModel(lr = 0.002, dropout_rate = .2, input_dropout = .2, mid_layer_size = 64, activation = 'sigmoid', image_size =40): # encapsulation to facilitate skopt usage, even though I didn't use it in the end.

    opt = Nadam(lr)



    

    model = Sequential()

    model.add(Dropout(input_dropout, input_shape = (image_size,image_size,3)))

    model.add(Conv2D(32, (3,3), activation = 'elu', padding = 'valid', 

                     

                    ))

 #   model.add(Dropout(dropout_rate))

    model.add(BatchNormalization())

    model.add(Conv2D(mid_layer_size, (3,3), activation = 'elu', padding = 'valid'))

 #   model.add(Dropout(dropout_rate))

    model.add(BatchNormalization())

    model.add(Conv2D(32, (3,3), activation = 'elu', padding = 'valid'))

  #  model.add(Dropout(dropout_rate))

    model.add(BatchNormalization())

    

    # I'll compute size of the dense layer:

    n= (image_size-6)*(image_size-6)*3

    model.add(Flatten())

    model.add(Dense(n, activation = 'elu'))

    model.add(Reshape((image_size-6, image_size-6, 3)))

    

    model.add(Conv2D(3, (5,5), activation = 'relu',  padding = 'valid'))

    

    n= (image_size-10)*(image_size-10)*3

    model.add(Flatten())

    model.add(Dense(n, activation = 'relu'))

    model.add(Dropout(dropout_rate))

    model.add(Dense(n, activation = 'relu'))

    model.add(Dropout(dropout_rate))

    model.add(Reshape((image_size-10, image_size-10, 3)))

    

    model.add(Conv2D(3, (5,5), activation = activation, padding = 'same', name = 'output_layer'))

    

    model.compile(loss = 'mean_absolute_error', optimizer = opt)

    return(model)



print(X_train.shape)

print(y_train.shape)
mae(X_train[:,5:image_size-5,5:image_size-5,:].reshape(-1,((image_size-10)*(image_size-10)*3)), y_train.reshape(-1,(image_size-10)*(image_size-10)*3)) # Because mse takes array of dim <=2
model = getModel(dropout_rate = .35, input_dropout = 0.0, image_size = image_size, mid_layer_size = 64)

model.summary() #to give overview of number of params

stop = EarlyStopping(patience=10, restore_best_weights = True)

model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_split = 0.2, callbacks = [stop], verbose = True)
mae(model.predict(X_train).reshape(-1,(image_size-2-2*n_convolutions)**2*3), y_train.reshape(-1,(image_size-2-2*n_convolutions)**2*3)) 
comp = plt.figure(figsize=(9, 13))

first = comp.add_subplot(3,1,1)

first.imshow(y_train[30])

second =comp.add_subplot(3,1,2)

second.imshow(X_train[30])

second =comp.add_subplot(3,1,3)

second.imshow(model.predict(np.array([X_train[30]]))[0])

comp.show()
#from skopt import gp_minimize

#from skopt.space import Real, Integer

#dropout_rate_space = Real(low = 0.0, high = 0.7)

#mid_layer_size_space = Integer(low = 16, high = 512)

#def f(v):

#    model = getModel(dropout_rate = v[0], mid_layer_size= v[1])

#    model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_split = 0.2, callbacks = [stop], verbose = False)

#    return(mse(model.predict(X_train).reshape(-1,(image_size-2-2*n_convolutions)**2*3), y_train.reshape(-1,(image_size-2-2*n_convolutions)**2*3)) )
#res = gp_minimize(f, [dropout_rate_space, mid_layer_size_space],

#                  n_calls = 50, n_random_starts = 6, verbose = True

#                 )

#print(res.v)