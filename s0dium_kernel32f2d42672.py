import zipfile

import os

import sys

import random





val_ratio = 0.1

train_dir = './train/'

test_dir  = './test/'

val_dir   = './val/'



def unzip():

    print('Unzipping...')

    

    with zipfile.ZipFile("../input/dogs-vs-cats-redux-kernels-edition/train.zip","r") as z:

        z.extractall("./")

        

    with zipfile.ZipFile("../input/dogs-vs-cats-redux-kernels-edition/test.zip","r") as z:

        z.extractall("./")

        

    print('Unzipped!')



def split_val_train():

    print('Splitting train / val...')

    try:

        os.makedirs(train_dir+'cat/')

        os.makedirs(train_dir+'dog/')

        os.makedirs(val_dir)

        os.makedirs(val_dir+'cat/')

        os.makedirs(val_dir+'dog/')

        os.makedirs(test_dir+'unknown/')

    except OSError:

        pass



    dogs = []

    cats = []

    

    for f in os.listdir(test_dir):

        if not(os.path.isdir(test_dir+f)):

            os.replace(test_dir + f, test_dir + 'unknown/' + f)



    for f in os.listdir(train_dir):

        s = f.split('.')

        if len(s) > 2:

            new_path = s[0] + '_' + s[1] + '.' + s[2]

            os.replace(train_dir + f, train_dir + s[0] + '/' + new_path)

            if s[0] == 'cat':

                cats.append(new_path)

            if s[0] == 'dog':

                dogs.append(new_path)



    print('Cur Dogs count:', len(dogs))

    print('Cur Cats count:', len(cats))



    dogs_val = random.sample(dogs, round(val_ratio*len(dogs)))

    cats_val = random.sample(cats, round(val_ratio*len(cats)))



    print('Target Dogs validation count:', len(dogs_val))

    print('Target Cats validation count:', len(cats_val))

    

    for dog in dogs_val:

        s = dog.split('.')

        if len(s) > 1:

            os.replace(train_dir+ 'dog/' + dog, val_dir + 'dog/' + dog)

        

    for cat in cats_val:

        s = cat.split('.')

        if len(s) > 1:

            os.replace(train_dir+ 'cat/' + cat, val_dir + 'cat/' + cat)

    

def prepare_data():

    if not(os.path.exists(train_dir)):

        unzip()

        

    if not(os.path.exists(val_dir)):

        split_val_train()

        

    print('** Data prepared. Check out: **')

        

    print('Dogs count:', len(os.listdir(train_dir+'dog')))

    print('Cats count:', len(os.listdir(train_dir+'cat')))



    print('Dogs validation count:', len(os.listdir(val_dir+'dog')))

    print('Cats validation count:', len(os.listdir(val_dir+'cat')))

    

    print('Test len:', len(os.listdir(test_dir+'unknown/')))



prepare_data()



import numpy as np

from tensorflow.python.keras.applications.resnet import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])



from tensorflow.python.keras.applications.resnet import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.callbacks import ModelCheckpoint



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



train_generator = data_generator.flow_from_directory(

        train_dir,

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

        val_dir,

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



# Создаем коллбек сохраняющий веса модели

checkpoint_path = "./cp.ckpt"

cp_callback = ModelCheckpoint(filepath=checkpoint_path,

                              save_weights_only=True,

                              verbose=1,

                              save_best_only=True)

doTrain = True



if doTrain:

    my_new_model.fit_generator(

            train_generator,

            steps_per_epoch=train_generator.n//24,

            epochs=4,

            validation_data=validation_generator,

            validation_steps=validation_generator.n//24,

            callbacks=[cp_callback])



    

my_new_model.load_weights(checkpoint_path)



test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(image_size, image_size),

    class_mode=None,

    shuffle=False,

    seed=42,

    batch_size=1)



#loss, acc = my_new_model.evaluate(test_generator, test_labels, verbose=2)

my_new_model.evaluate_generator(generator=validation_generator, steps=validation_generator.n//32)



test_generator.reset()



pred = my_new_model.predict_generator(test_generator, steps=10, verbose=1)



predicted_class_indices=np.argmax(pred,axis=1)



print(predicted_class_indices[0:10])



print(test_generator.filenames[0:10])