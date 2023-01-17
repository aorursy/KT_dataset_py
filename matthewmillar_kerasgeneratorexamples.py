import os

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications.xception import preprocess_input

import matplotlib.pyplot as plt

import numpy as np
DATA_PATH = '../input/gtsplitfolders/Data/'
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    shear_range=0,

    rotation_range=20,

    zoom_range=0.15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    fill_mode = 'nearest')



train_generator = train_datagen.flow_from_directory(

    DATA_PATH,

    target_size=(224, 224),

    batch_size=BATCH_SIZE,

    shuffle = True,

    class_mode='categorical')
def tripleloss_generator(generator):

    while True:

        img, label = next(generator)

        train_set = [img, label]

        yield train_set, label
trip_gen = tripleloss_generator(train_generator)

X, label =next(trip_gen)

print(label)

print(X[0].shape)

img = np.squeeze(X[0][0])

plt.imshow(img)
anchor_generator = train_datagen.flow_from_directory(

    DATA_PATH,

    target_size=(224, 224),

    batch_size=BATCH_SIZE,

    shuffle = True,

    class_mode='categorical')

other_generator = train_datagen.flow_from_directory(

    DATA_PATH,

    target_size=(224, 224),

    batch_size=BATCH_SIZE,

    shuffle = True,

    class_mode='categorical')
def pairwise_generator(anchor_gen, other_gen):

    while True:

        a_img, a_label = next(anchor_gen)

        o_img, o_label = next(other_gen)

        

        if np.argmax(a_label) == np.argmax(o_label):

            bi_label = 1

        else:

            bi_label = 0

        yield [a_img, o_img], [a_label, o_label, bi_label]

    
pair_gen = pairwise_generator(anchor_generator, other_generator)

imgs, labels = next(pair_gen)

a_lable = np.argmax(labels[0])

o_lable = np.argmax(labels[1])

bi_lable = np.argmax(labels[2])



print(a_lable)

print(o_lable)

print(bi_lable)

anchor = imgs[0][0]

other = imgs[1][0]

anchor = np.squeeze(anchor)

other = np.squeeze(other)

f, axarr = plt.subplots(1,2)

axarr[0].imshow(anchor)

axarr[1].imshow(other)



# Takes in a list (numpy array of image files)

def process_image(count, path):

    if count % 2 == 0:

        img = preprocess_img(path, (300,300))

        label = 0

    else:

        img = preprocess_img(path, (300,300))

        img, label = alter_image(img)

    print(label)

    return img, label

        

        

def data_generator(files, number_classes, batch_size = 32):

    while True:

        batch_paths = np.random.choice(a = files, size = batch_size)

        batch_inputs = []

        batch_outputs = []

        count = 1

        for input_path in batch_paths:

            print(input_path)

            img, label = process_image(count, input_path)

            label = to_categorical(label, num_classes=number_classes)

            batch_inputs.append(img)

            batch_outputs.append(label)

            count += 1

        batch_x = np.array(batch_inputs)

        batch_y = np.array(batch_outputs)

        yield batch_x, batch_y

            

        

# gen = data_generator(img_array_data, 5, 5)

# x, y = next(iter(gen))

# print(y)   

        
#Same as above but with two lables outputs

def process_image(count, path):

    if count % 2 == 0:

        img = preprocess_img(path, (300,300))

        class_label = 0

        good_label = 0

    else:

        img = preprocess_img(path, (300,300))

        img, class_label = alter_image(img)

        good_label = 1

        

    return img, class_label, good_label

        

        

def data_generator(files, number_classes, batch_size = 32):

    while True:

        batch_paths = np.random.choice(a = files, size = batch_size)

        batch_inputs = []

        batch_out_bi = []

        batch_outputs_class = []

        count = 1

        for input_path in batch_paths:

            img, class_label, good_label = process_image(count, input_path)

            class_label = to_categorical(class_label, num_classes=number_classes)

            good_label = to_categorical(good_label, num_classes=2)

            batch_inputs.append(img)

            batch_out_bi.append(good_label)

            batch_outputs_class.append(class_label)

            

            count += 1

        batch_x = np.array(batch_inputs)

        batch_y_bi = np.array(batch_out_bi)

        batch_y_class = np.array(batch_outputs_class)

        #print("X:{} Bi:{} Y:{}".format(batch_x.shape, batch_y_bi.shape, batch_y_class.shape))

        yield batch_x, [batch_y_bi, batch_y_class]
######

#Generate good and bad image

#####

import os

import matplotlib.pyplot as plt

import numpy as np

import random

from PIL import Image





def preprocessing_image(path, size):

    # Read in image

    img = Image.open(path)

    # Resize to new shape

    img.thumbnail(size)

    # img /= 255 # Normalize image

    return img



def random_rotator(img):

    rot = random.randint(-45, 45)

    img = img.rotate(rot)

    return img



def crop_image(img):

    left = 100

    top = 100

    right = 400

    bottom = 400

    img = img.crop((left, top, right, bottom))

    img.thumbnail((224, 224))  # Resize the cropped image to vgg19 defualt size

    return img



def augment_image(path, size, augment=True):

    img = preprocessing_image(path, size=size)

    # Only rotate and crop image

    if augment:

        img = random_rotator(img)

        img = crop_image(img)

        return img

    else:

        # Other wise dont rotate it

        img = crop_image(img)

        return img



def create_image_array(datapath):

    images = []

    for file in os.listdir(datapath):

        join = os.path.join(datapath, file)

        images.append(join)

    return images



def data_generator(files, size, batch_size=32):

    while True:

        batch_paths = np.random.choice(a=files, size=batch_size)

        batch_inputs = []

        batch_outputs = []

        count = 0

        for input_path in batch_paths:

            if count % 2 == 0:

                aug_img = True

                label = [1, 0]

            else:

                aug_img = False

                label = [0, 1]

            count += 1

            img = augment_image(input_path, size, aug_img)

            img = np.array(img)

            batch_inputs.append(img)

            batch_outputs.append(label)



        batch_x = np.array(batch_inputs)

        batch_y = np.array(batch_outputs)

        yield batch_x, batch_y



        

'''

my_array = create_image_array(DATA_PATH)



    gen = data_generator(my_array, (600, 600), 5)



    x, y = next(gen)

    print(x.shape)

    for i in range(len(x)):

        x1 = x[i]

        y1 = y[i]

        x1 = np.squeeze(x1)

        print(y1)

        plt.imshow(x1)

        plt.show()

'''