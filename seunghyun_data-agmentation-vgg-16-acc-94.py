# Old version : from keras.~ import ~



from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.layers import Input

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



import glob # Extract specific files

import os

import pandas as pd

import numpy as np

from numpy import expand_dims

import matplotlib.pyplot as plt

%matplotlib inline



from pathlib import Path

from skimage.io import imread
# seed

np.random.seed(111)



# data_dir

train_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train"

test_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test"

val_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val"



# Data Size

print("Train Set : %d" %(len(os.listdir(train_dir + "/PNEUMONIA")) + len(os.listdir(train_dir + "/NORMAL"))))

print("Validation Set : %d" %(len(os.listdir(val_dir + "/PNEUMONIA")) + len(os.listdir(val_dir + "/NORMAL"))))

print("Test Set : %d" %(len(os.listdir(test_dir + "/PNEUMONIA")) + len(os.listdir(test_dir + "/NORMAL"))))
normal_dir = Path(train_dir + '/NORMAL')

pneumonia_dir = Path(train_dir + '/PNEUMONIA')



normal = normal_dir.glob('*.jpeg')

pneumonia = pneumonia_dir.glob('*.jpeg')





train_n, train_p = [], []

i, j = 0, 0

for img_n in normal:

    train_n.append(img_n)

    i += 1

    if i == 4 :

        break

for img_p in pneumonia:

    train_p.append(img_p)

    j += 1

    if j == 4 :

        break



# Comparison

samples = train_n + train_p



f, ax = plt.subplots(2,4, figsize=(30,10))

for i in range(8):

    img = imread(samples[i])

    ax[i//4, i%4].imshow(img, cmap='gray')

    if i<4:

        ax[i//4, i%4].set_title("Pneumonia",size = 30)

    else:

        ax[i//4, i%4].set_title("Normal",size = 30)

    ax[i//4, i%4].axis('off')

    ax[i//4, i%4].set_aspect('auto')

plt.show()

train_datagen = ImageDataGenerator(

    rescale = 1./255,

)



training_set = train_datagen.flow_from_directory(

    train_dir,

    target_size = (150, 150), # image size

    batch_size = 2,

    class_mode = 'binary'

)



img_name = 'person63_bacteria_306.jpeg'

img_pneumonia = load_img(train_dir + '/PNEUMONIA/' + img_name)



plt.imshow(img_pneumonia)

plt.show()
data = img_to_array(img_pneumonia)

samples = expand_dims(data, 0)



datagen = ImageDataGenerator(

    rescale = 1./255,

    rotation_range=15,

    shear_range = 0.3, 

    zoom_range =0.3, 

    height_shift_range=0.1,

    width_shift_range=0.1,

    horizontal_flip = True

)



it = datagen.flow(samples, batch_size = 1)

fig = plt.figure(figsize = (10,10))

for i in range(9):

    plt.subplot(3,3, i+1)    

    batch = it.next()    

    image = batch[0]

    plt.imshow(image)

nb_train_samples, nb_val_samples = 5218, 18

img_width, img_height = 160, 160

batch_size = 16





train_datagen = ImageDataGenerator(

rescale = 1./255, 

    shear_range = 0.3, 

    zoom_range = 0.3,

    height_shift_range=0.1,

    width_shift_range=0.1,

    horizontal_flip = True

)



training_set = train_datagen.flow_from_directory(

    train_dir,

    target_size = (img_width, img_height), 

    batch_size = batch_size,

    class_mode = 'binary'

)

test_datagen = ImageDataGenerator( rescale = 1./255 )



validation_set = test_datagen.flow_from_directory(

    val_dir,

    target_size = (img_width, img_height),

    batch_size = batch_size,

    class_mode = 'binary'

)



testing_set = test_datagen.flow_from_directory(

    test_dir,

    target_size = (img_width, img_height),

    batch_size = batch_size,

    class_mode = 'binary'

)
###### VGG16 ######

def vgg_m():

    input_data = Input(shape=(img_width, img_height, 3), name = "InputData")



    # (1)

    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu')(input_data)

    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = MaxPooling2D((2,2))(x)

    

    # (2)

    x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = MaxPooling2D((2,2))(x)

    

    # (3)

    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = MaxPooling2D((2,2))(x)

    

    #(4)

    x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = MaxPooling2D((2,2))(x)

    

    #(5)

    x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)

    x = MaxPooling2D((2,2))(x)

    

    # (6)

    x = Flatten()(x)

    x = Dense(4096, activation = 'relu')(x)

    x = Dropout(0.3)(x) # 과적합 방지 위해

    x = Dense(4096, activation = 'relu')(x)

    x = Dropout(0.3)(x)

    x = Dense(1024, activation = 'relu')(x)

    x = Dropout(0.3)(x)

    output = Dense(1, activation = 'sigmoid')(x) # Because of binomial classification

    

    model = Model(input_data, output)

    

    return model
vgg = vgg_m()

vgg.summary()
# When I used Adam, the model fell into the local minima

vgg.compile(optimizer = RMSprop(lr=0.00005), loss = 'binary_crossentropy', metrics = ['acc'])



early_stop = EarlyStopping(monitor= 'loss', patience = 3, verbose = 1)





checkpoint = ModelCheckpoint(filepath='model.hdf5', 

                             monitor='loss',

                             mode='min',

                             save_best_only=True)

history = vgg.fit_generator( training_set,

                            steps_per_epoch = nb_train_samples // batch_size,                               

                            epochs = 50,                              

                            validation_data = validation_set,

                            callbacks = [early_stop, checkpoint]

                           )



#vgg_model.save('D:/chest-xray/model/'+'term_prj.h5')
accu = vgg.evaluate_generator(testing_set,steps=624)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

 

epochs = range(1, len(acc) + 1)

 

plt.plot(epochs, acc, 'b', label='Training acc')

plt.title('Accuracy')

plt.legend()

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.title('Loss')

plt.legend()

plt.show()



accu = vgg.evaluate_generator(testing_set,steps=624)

print('The testing accuracy is :',accu[1]*100, '%')
