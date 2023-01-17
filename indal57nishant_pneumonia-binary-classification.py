# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Softmax,Input,Flatten

from keras.optimizers import Adam,RMSprop,SGD

from keras.layers.merge import add

from keras.layers import Dense, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import BatchNormalization

from math import ceil









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



#attempt to fix reproducibility issue (hard to get reproducible results even after setting seed)

from tensorflow import set_random_seed

os.environ['PYTHONHASHSEED'] = "0"

np.random.seed(1)

set_random_seed(2)



print(os.listdir("../input/chest_xray/chest_xray"))



# Any results you write to the current directory are saved as output.
auggen = ImageDataGenerator(

#         rotation_range=40,

#         width_shift_range=0.1,

#         height_shift_range=0.1,

# #         shear_range=0.2,

# #         zoom_range=0.2,

#         horizontal_flip=True,

#         vertical_flip=True,

        rescale=1./255

        )

auggen = auggen.flow_from_directory(directory="../input/chest_xray/chest_xray/train",

                                    target_size=(256, 256), color_mode='rgb',  class_mode='binary', 

         batch_size=32, shuffle=True, seed=1)
auggen.class_indices
i=0

k=0

fig,axis1 = plt.subplots(1,10,figsize=(60,60))



#augggen.flow_from_directory returns a DirectoryIterator yielding tuples of (x, y) for each iteration where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.

#here we unpack the tuple into the first element, images, which is of dimension (batch_size,length, width, number of color channels)

#think of the number of iterations that the DirectoryIterator object undergoes as the steps_per_epoch parameter in the fit method



#if we set class_mode= categorical then the labels will be hot encoded, if we set class_mode=binary then it will be a 1d array

#outer loop is used to iterate ove auggen (generator object)

for images,labels in  auggen:

    print(images.shape)

    print(labels[0])

    

    #iterating over images in the first batch in order to plot

    for image in images:

        axis1[k].imshow(image)

        axis1[k].set_title(labels[k],fontdict={'fontsize':50})

        k=k+1

        #I only want to plot the first 10 images but we have all 32 available in images variable after 1st iteration

        if k==10:

            break

    

    i=i+1

    if i==1:

        break

    
print('total number of positive instances (pneumonia) in first batch of 32: {}'.format(sum(labels)))
traingen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        rescale=1./255

        )

testgen = ImageDataGenerator(

        rescale=1./255

        )



valgen = ImageDataGenerator(

        rescale=1./255

        )



traingen = traingen.flow_from_directory(directory="../input/chest_xray/chest_xray/train", 

    target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=32, shuffle=True, seed=1)

        

                                   



                                   

                                                                      

                                   
testgen = testgen.flow_from_directory(directory="../input/chest_xray/chest_xray/test", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=624, shuffle=False)

                                   
valgen = valgen.flow_from_directory(directory="../input/chest_xray/chest_xray/val", 

                                      target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=16, shuffle=False)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same",

                 input_shape=(256,256,1)))

model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))

# model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))

# model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(1024,activation="relu"))

# model.add(BatchNormalization())

# model.add(Dropout(rate=0.4))

model.add(Dense(1, activation="sigmoid"))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['binary_accuracy'])
model.summary()
history=model.fit_generator(

        traingen,

    

        steps_per_epoch=5216/32,

        validation_data=valgen,

        validation_steps=4,

        epochs=3,

        class_weight = {0:2.94,

                        1:1}

)
len(testgen)
model.evaluate_generator(testgen, steps = len(testgen))

y_pred=model.predict_generator(testgen, steps = len(testgen))

model.save("pneumonia_classifier.h5")
len(y_pred)


from keras.preprocessing import image

import matplotlib.pyplot as plt



x,y = testgen.next()

print(y)

print(sum(y))

print(len(y))
len(testgen)
history.history

traingen_augment = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        rescale=1./255

        )

# testgen2 = ImageDataGenerator(

#         rescale=1./255

#         )



# valgen2 = ImageDataGenerator(

#         rescale=1./255

#         )
traingen_augment = traingen_augment.flow_from_directory(directory="../input/chest_xray/chest_xray/train/NORMAL", 

    target_size=(256, 256), color_mode='grayscale',  class_mode='binary', 

         batch_size=32, shuffle=True, seed=1,save_to_dir=None, save_prefix='augmented', save_format='jpeg')

        