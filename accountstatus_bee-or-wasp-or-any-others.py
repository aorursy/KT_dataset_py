import pandas as pd

import numpy as np
import os

from random import shuffle

import shutil

import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing import image as img

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Let's create the directories for training and validation

main_path='./output/bee-vs-wasp'

training_path=main_path+'/training'

testing_path=main_path+'/validation'

bee_training=training_path+'/bee'

wasp_training=training_path+'/wasp'

other_insects_training=training_path+'/other_insects'

other_noinsects_training=training_path+'/other_noinsects'

bee_testing=testing_path+'/bee'

wasp_testing=testing_path+'/wasp'

other_insects_testing=testing_path+'/other_insects'

other_noinsects_testing=testing_path+'/other_noinsects'
def directory_creation(path):

    os.mkdir(path)
directory_creation('./output')
directory_creation(main_path)

directory_creation(training_path)

directory_creation(testing_path)

directory_creation(bee_training)

directory_creation(wasp_training)

directory_creation(other_insects_training)

directory_creation(other_noinsects_training)
directory_creation(bee_testing)

directory_creation(wasp_testing)

directory_creation(other_insects_testing)

directory_creation(other_noinsects_testing)
def copy_files(original,dest_training,dest_testing):

    l=os.listdir(original)

    training_length=int(len(l)*0.8)

    shuffle(l)

    for i in range(len(l)):

        if i<training_length:

            shutil.copy(original+'/'+l[i],dest_training)

        else:

            shutil.copy(original+'/'+l[i],dest_testing)

            
copy_files('../input/bee-vs-wasp/kaggle_bee_vs_wasp/bee1',bee_training,bee_testing)
copy_files('../input/bee-vs-wasp/kaggle_bee_vs_wasp/bee2',bee_training,bee_testing)
copy_files('../input/bee-vs-wasp/kaggle_bee_vs_wasp/wasp1',wasp_training,wasp_testing)
copy_files('../input/bee-vs-wasp/kaggle_bee_vs_wasp/wasp2',wasp_training,wasp_testing)
copy_files('../input/bee-vs-wasp/kaggle_bee_vs_wasp/other_insect',other_insects_training,other_insects_testing)
copy_files('../input/bee-vs-wasp/kaggle_bee_vs_wasp/other_noinsect',other_noinsects_training,other_noinsects_testing)
data_generator = ImageDataGenerator(rescale = 1./250,zoom_range = 0.2)

batch_size = 8 #accessing all our data both training and testing

training_data = data_generator.flow_from_directory(directory = training_path,

                                                  target_size = (150,150),

                                                  batch_size = batch_size,)

testing_data = data_generator.flow_from_directory(directory = testing_path,

                                                  target_size = (150,150),

                                                  batch_size = batch_size)
model = Sequential() #making our CNN

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = training_data.image_shape))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.3))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 126, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.15))

model.add(Flatten())

model.add(Dense(units = 32, activation = 'relu'))

model.add(Dropout(rate = 0.15))

model.add(Dense(units = 64, activation = 'relu'))

model.add(Dropout(rate = 0))

model.add(Dense(units = len(set(training_data.classes)), activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

fitted_model = model.fit_generator(training_data,

                        steps_per_epoch = 250,

                        epochs = 15,

                        validation_data = testing_data,

                        validation_steps = 1000)
# now let's save the model so we dont have to train this again

model.save('./output/model')
def testing_image(image_directory): #testing out our model

    test_image = img.load_img(image_directory, target_size = (150, 150))

    test_image = img.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis = 0)

    result = model.predict(x = test_image)

    print(result)

    if result[0][2]==max(result[0]):

        print('Other than insects')

    elif result[0][1]==max(result[0]):

        print('Insects')

    elif result[0][0]==max(result[0]):

        print('Bees')

    else:

        print('oooo its the wasps')
from IPython.display import Image

Image("../input/bee-vs-wasp/kaggle_bee_vs_wasp/bee1/10092043833_7306dfd1f0_n.jpg")
testing_image('../input/bee-vs-wasp/kaggle_bee_vs_wasp/bee1/10092043833_7306dfd1f0_n.jpg')
# let's try the insects now

Image('../input/bee-vs-wasp/kaggle_bee_vs_wasp/other_insect/10199076566_2014fdb8a8_n.jpg')
testing_image('../input/bee-vs-wasp/kaggle_bee_vs_wasp/other_insect/10199076566_2014fdb8a8_n.jpg')
# other objects other than the insects

Image('../input/bee-vs-wasp/kaggle_bee_vs_wasp/other_noinsect/501094.jpg')
testing_image('../input/bee-vs-wasp/kaggle_bee_vs_wasp/other_noinsect/501094.jpg')
# Wasps Which looks closer to the bees

Image('../input/bee-vs-wasp/kaggle_bee_vs_wasp/wasp2/G00017.jpg')
testing_image('../input/bee-vs-wasp/kaggle_bee_vs_wasp/wasp2/G00017.jpg')
# So we can see we are getting good results 

# Thanks for givin the file a read :)