from piltonumpy_helper import pil_to_numpy

import os; import keras; import numpy as np; import pandas as pd; import shutil; from PIL import Image; import cv2

from tqdm import tqdm

from skimage.transform import resize, rescale



import matplotlib.pyplot as plt; import seaborn as sns;



from keras.layers import *

from keras.models import *



from tensorflow.keras.layers import Add

from keras.preprocessing.image import ImageDataGenerator
DATA_DIR = '/kaggle/input/the-car-connection-picture-dataset'

LOW_RES = '/kaggle/working/train_lowres'
os.mkdir('/kaggle/working/train_lowres')
os.chdir('/kaggle/working/train_lowres')
files = os.listdir(DATA_DIR)
len(files)
files.sort()
import multiprocessing

from multiprocessing import Process


def make_low_res(files):

    for file in tqdm(files):

        img = Image.open(DATA_DIR + '/' + file)

        img = pil_to_numpy(img)

        #img = img/255.0

        img = cv2.resize(img, (256, 256))

        img = cv2.resize(cv2.resize(img, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC), None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

        im = Image.fromarray(img)

        im.save(file)

    
import time
# Making the low resolution folder in parallel processing



start = time.time()



p1 = Process(target = make_low_res, args = (files[:15000],))

p2 = Process(target = make_low_res, args = (files[15000:30000],))

p3 = Process(target = make_low_res, args = (files[30000:45000],))

p4 = Process(target = make_low_res, args = (files[45000:60000],))

p5 = Process(target = make_low_res, args = (files[60000:],))



p1.start()

p2.start()

p3.start()

p4.start()

p5.start()



p1.join()

p2.join()

p3.join()

p4.join()

p5.join()



end = time.time()



print('time elapsed -> ', end - start)
len(os.listdir('/kaggle/working/train_lowres'))




def fetch_data_generator(files, batch_size = 64):

    while True:

        #

        batch_files = np.random.choice(files, batch_size)

        

        batch_x = [] ; batch_y = [];

        

        for file in batch_files:

            img = cv2.resize(pil_to_numpy(Image.open(DATA_DIR + '/' + file)).astype(float), (256,256))

            img_low = pil_to_numpy(Image.open(LOW_RES + '/' + file)).astype(float)

            

            batch_x.append(img_low/255.0)

            batch_y.append(img/255.0)

        

                

        yield np.array(batch_x), np.array(batch_y)
"""encoder = Sequential()

encoder.add(Conv2D(64, (3,3) , padding = 'same', activation = 'relu', input_shape = (256, 256, 3)))

encoder.add(Conv2D(64, (3,3), padding = 'same', activation = 'relu'))

encoder.add(MaxPooling2D((2,2), padding = 'same'))

encoder.add(Dropout(0.3))

encoder.add(Conv2D(128, (3,3), padding = 'same', activation = 'relu'))

encoder.add(Conv2D(128, (3,3), padding = 'same', activation = 'relu'))

encoder.add(MaxPooling2D((2,2), padding = 'same'))

encoder.add(Conv2D(256, (3,3), padding = 'same', activation = 'relu'))

encoder.summary()"""
#autoencoder = encoder
#autoencoder.summary()
# encoder



keras.backend.set_image_data_format('channels_last')



i1 = Input(shape = (256,256,3))

l1 = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(i1)

l2 = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(l1)

l3 = MaxPooling2D(padding = 'same')(l2)

l3 = Dropout(0.3)(l3)

l4 = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(l3)

l5 = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(l4)

l6 = MaxPooling2D(padding = 'same')(l5)

l7 = Conv2D(256, (3,3), padding = 'same', activation = 'relu')(l6)



# decoder



l8 = UpSampling2D()(l7)

l9 = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(l8)

l10 = Conv2D(128, (3,3), padding = 'same', activation = 'relu')(l9)

l11 = Add()([l5, l10])

l12 = UpSampling2D()(l11)

l13 = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(l12)

l14 = Conv2D(64, (3,3), padding = 'same', activation = 'relu')(l13)

l15 = Add()([l14, l2])



# final layer should have 3 channels which will help to reconstruct the image with better resolution

l16 = Conv2D(3, (3,3), padding = 'same', activation = 'relu')(l15)



autoencoder = Model(i1, l16)
autoencoder.summary()
autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')
autoencoder.fit_generator(fetch_data_generator(files, 32), steps_per_epoch = len(files)//32, epochs = 10)
autoencoder.save('trained_10epochs.h5')
# Load Model

# uncomment to load the model with supplied weights & config, i.e., hdf5



#trained_model = keras.models.load_model('my_model.h5')