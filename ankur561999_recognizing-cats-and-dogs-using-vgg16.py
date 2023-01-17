# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from PIL import Image

from keras.models import Sequential

import keras, shutil

from keras.layers import Conv2D, ZeroPadding2D,Flatten,Dense, MaxPooling2D

from zipfile import ZipFile

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam
# extract train.zip

zip_train_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"

with ZipFile(zip_train_path) as myzip:

    myzip.extractall('/kaggle/temp/') 
# extract test.zip

zip_test_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"

with ZipFile(zip_test_path) as myzip:

    myzip.extractall('/kaggle/temp/')
files = os.listdir('/kaggle/temp/train/') 
# view sample submission

sample_submission = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

sample_submission.head()
test_files = os.listdir('/kaggle/temp/test/')

test_files[0:5]
# create sub directory cat and dog in train

os.mkdir('/kaggle/temp/train/cat') 

os.mkdir('/kaggle/temp/train/dog') 
# create validation set directory and sub directories cat and dog

os.mkdir('/kaggle/temp/val') 

os.mkdir('/kaggle/temp/val/cat') 

os.mkdir('/kaggle/temp/val/dog') 
# move cat images to train/cat

# and dog images to train/dog

images_path = '/kaggle/temp/train/'

for file in files[0:20000]:

    if file[0] == 'd':

        dst = '/kaggle/temp/train/dog/'

        img = os.path.join(images_path, file) 

        shutil.move(img, dst) 

    elif file[0] == 'c':

        dst = '/kaggle/temp/train/cat/'

        img = os.path.join(images_path, file) 

        shutil.move(img, dst) 

        

# rest of files move to validation directory

for file in files[20000:25000]:

    if file[0] == 'd':

        dst = '/kaggle/temp/val/dog/'

        img = os.path.join(images_path, file) 

        shutil.move(img, dst) 

    elif file[0] == 'c':

        dst = '/kaggle/temp/val/cat/'

        img = os.path.join(images_path, file) 

        shutil.move(img, dst) 

        

# 
# print num of samples in training and validation directory

train_num_samples = 0

for _,_,filenames in os.walk('/kaggle/temp/train/'):

    train_num_samples += len(filenames) 

    

print('train_num_samples: ', train_num_samples) 



val_num_samples = 0

for _, _, filenames in os.walk('/kaggle/temp/val/'):

    val_num_samples += len(filenames) 

    

print('val_num_samples: ', val_num_samples) 
# create Image Data Generator

datagen = ImageDataGenerator(rescale = 1.0/255) 



train_batch_size = 16

train_dir = '/kaggle/temp/train/'

# create train generator

train_gen = datagen.flow_from_directory(train_dir, 

                                        target_size = (224, 224), 

                                        batch_size = train_batch_size, 

                                        class_mode = 'binary') 



val_batch_size = 16

val_dir = '/kaggle/temp/val/'

val_gen = datagen.flow_from_directory(val_dir, 

                                      target_size = (224, 224), 

                                      batch_size = val_batch_size, 

                                      class_mode = 'binary') 
print(train_gen.class_indices) 
x, y = train_gen.next()

plt.figure(figsize = (13,13))



for i,(img, label) in enumerate(zip(x, y)):

    plt.subplot(4, 4,i+1) 

    if label == 1:

        plt.title('Dog') 

    else:

        plt.title('Cat') 

        

    plt.axis('off') 

    plt.imshow(img) 
n_h = 224 # height of the image

n_w = 224 # width of the image

n_c = 3 # num of channels in image



# num of filters for convolutional layers

n_filters = [64,64,128,128,256,256,256,512,512,512,512,512,512]

len(n_filters) 
model = Sequential()



# create 2 conv layers with 64 filters of size 3,padding same and stride of 1

model.add(Conv2D(filters = n_filters[0], input_shape = (n_w, n_h, n_c), kernel_size = 3,

                 padding = 'SAME', activation='relu')) 

model.add(Conv2D(filters = n_filters[1],kernel_size = 3,

                 padding = 'SAME', activation='relu'))



# create max pooling layer

model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))



# create 2 conv layers with 128 filters

model.add(Conv2D(filters = n_filters[2],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

model.add(Conv2D(filters = n_filters[3],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

# create max pooling layer

model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2))) 



# create 3 conv layers with 256 filters

model.add(Conv2D(filters = n_filters[4],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

model.add(Conv2D(filters = n_filters[5],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

model.add(Conv2D(filters = n_filters[6],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

# create max pooling layer

model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2))) 



# create 3 conv layers with 512 filters



model.add(Conv2D(filters = n_filters[7],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

model.add(Conv2D(filters = n_filters[8],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

model.add(Conv2D(filters = n_filters[9],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

# create max pooling layer

model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))



# create 3 conv layers with 512 filters



model.add(Conv2D(filters = n_filters[10],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

model.add(Conv2D(filters = n_filters[11],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

model.add(Conv2D(filters = n_filters[12],kernel_size = 3,

                 padding = 'SAME', activation='relu'))

# create max pooling layer

model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))



# flatten the layer

model.add(Flatten())



# create 2 fully connected layers with 4096 neurons

model.add(Dense(units = 4096, activation='relu')) 

model.add(Dense(units = 4096, activation='relu')) 



# final add the output layer

model.add(Dense(units = 1, activation = 'sigmoid')) 





# print model summary

model.summary()
# compile the model

model.compile(loss='binary_crossentropy', 

              optimizer = Adam(learning_rate = 0.00003), 

              metrics = ['accuracy']) 



# calculate train_steps and val_steps

train_steps = np.ceil(train_num_samples / train_batch_size) 

val_steps = np.ceil(val_num_samples / val_batch_size) 



# train the model

history = model.fit_generator(train_gen, 

                              steps_per_epoch = train_steps, 

                              epochs = 12,

                              validation_data = val_gen, 

                              validation_steps = val_steps) 
model.save('/kaggle/working/recognizing_cats_and_dogs_using_vgg16.h5') 
# load the model

model = keras.models.load_model("/kaggle/working/recognizing_cats_and_dogs_using_vgg16.h5")



test_path = "/kaggle/temp/test/"



# create sub directory all_data inside test directory

os.mkdir("/kaggle/temp/test/all_data")



# move all test images to all_data

for file in test_files:

    img = os.path.join(test_path, file)

    dst = "/kaggle/temp/test/all_data/"

    shutil.move(img, dst)



# create test generator

test_gen = datagen.flow_from_directory(test_path,

                                      target_size = (224, 224),

                                      batch_size = 16,

                                      class_mode = None,

                                      shuffle = False)

# make predictions

preds = model.predict_generator(test_gen, 

                               steps = len(test_gen))

preds = np.squeeze(preds)



# create the dataframe

submission = pd.DataFrame({"files": test_files, "label": preds})

submission.head()
submission['label'] = submission['label'].round(3)

submission.head()
submission['files'] = submission['files'].str.replace('.jpg', '')

submission['files'] = submission['files'].astype('int')

submission.sort_values(['files'], ascending=True, inplace = True)

submission.head()
submission = submission.rename(columns = {'files': 'id'})

submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()