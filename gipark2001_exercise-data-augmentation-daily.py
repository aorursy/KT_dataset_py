from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



my_new_model.layers[0].trainable = False



my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.exercise_5 import *

print("Setup Complete")
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224



# Specify the values for all arguments to data_generator_with_aug.

data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                              horizontal_flip = True,

                                              width_shift_range = 0.1,

                                              height_shift_range = 0.1)

            

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

q_1.solution()
q_2.solution()
# Specify which type of ImageDataGenerator above is to load in training data

train_generator = data_generator_with_aug.flow_from_directory(

        directory = '../input/dogs-gone-sideways/images/train',

        target_size=(image_size, image_size),

        batch_size=12,

        class_mode='categorical')



# Specify which type of ImageDataGenerator above is to load in validation data

validation_generator = data_generator_no_aug.flow_from_directory(

        directory = '../input/dogs-gone-sideways/images/val',

        target_size=(image_size, image_size),

        class_mode='categorical')



my_new_model.fit_generator(

        train_generator, # if you don't know what argument goes first, try the hint

        epochs = 3,

        steps_per_epoch=19,

        validation_data=validation_generator)

q_3.check()
# q_3.hint()

# q_3.solution()
#q_4.solution()