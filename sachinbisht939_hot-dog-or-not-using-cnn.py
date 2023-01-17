import os

train_path = '../input/hot-dog-not-hot-dog/seefood/train'
test_path = '../input/hot-dog-not-hot-dog/seefood/test'

hot_dog_path = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'
not_hot_dog_path = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'
train_data_hd = [os.path.join(hot_dog_path, filename)
              for filename in os.listdir(hot_dog_path)]
train_data_nhd = [os.path.join(not_hot_dog_path, filename)
              for filename in os.listdir(not_hot_dog_path)]


import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

img_size = 224
num_classes = 2

data_generator = ImageDataGenerator()

train_generator = data_generator.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=498,
        class_mode='categorical')

# print(len(train_generator))
# print(len(train_generator[0]))
# print(train_generator[0][1])
# print(train_generator[0][0].shape)

validation_generator = data_generator.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=500,
        class_mode='categorical')

# print(len(validation_generator))
# print(len(validation_generator[0]))
# print(validation_generator[0][0].shape)


# TO DISPLAY IMAGE FROM DATA
# from IPython.display import Image, display
# display(Image(train_data_nhd[1]))

from tensorflow.python.keras.applications.resnet50 import preprocess_input

def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)

train_data = read_and_prep_images(train_data_nhd)
train_data1 = read_and_prep_images(train_data_hd)

# from IPython.display import Image, display
# display(Image(train_data_hd[1]))

from matplotlib import pyplot as plt

data = train_generator[0][0][3]
plt.imshow(data, interpolation='nearest')
plt.show()

from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout
import itertools
# import sys

model = Sequential()
model.add(Conv2D(8, kernel_size=(3,3),
#             strides=2,
            activation='relu',
            input_shape=(img_size, img_size, 3)))
Dropout(.5)
model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))
Dropout(.5)

model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))
model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))
model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))
model.add(Conv2D(8, kernel_size=(3,3),
#                 strides=2,
                activation='relu'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer='adam',
             metrics=['accuracy'])

input_x = (train_generator[0][0]/255)
input_y = (train_generator[0][1])

model.fit(input_x,
         input_y,
         batch_size=24,
         epochs=5)

# model.fit_generator(train_generator,
#         steps_per_epoch=5,
#         validation_data=validation_generator,
#         validation_steps=1)

output_x = (validation_generator[0][0]/255)
output_y = validation_generator[0][1]

print(model.evaluate(output_x, output_y))

pre1 = model.predict(train_data1)
print("Hot Dog")
for i in range(10):
    #display(Image(train_data_hd[i]))
#     print(pre1[i][0], pre1[i][1])
    print(np.argmax(pre1[i]))

pre = model.predict(train_data)
print("Not Hot Dog")
for i in range(10):
    #display(Image(train_data_nhd[i]))
    print(np.argmax(pre[i]))


from tensorflow.python.keras.applications import ResNet50
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
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
# The ImageDataGenerator was previously generated with
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# recent changes in keras require that we use the following instead:
data_generator = ImageDataGenerator() 

train_generator = data_generator.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        test_path,
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)