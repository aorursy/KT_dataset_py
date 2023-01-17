
from subprocess import check_output
from zipfile import ZipFile
print(check_output(["ls", "../input/python-utility-code-for-deep-learning-exercises"]).decode("utf8"))
zip_path = '../input/python-utility-code-for-deep-learning-exercises/utils'
with ZipFile(zip_path) as myzip:
    files_in_zip = myzip.namelist()
    print(files_in_zip)

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# num_classes is the number of categories your model chooses between for each prediction
num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# The value below is either True or False.  If you choose the wrong answer, your modeling results
# won't be very good.  Recall whether the first layer should be trained/changed or not.
my_new_model.layers[0].trainable = False
# We are calling the compile command for some python object. 
# Which python object is being compiled? Fill in the answer so the compile command works.
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocess_input)

train_generator = data_generator.flow_from_directory(
       directory = "../input/dogs-gone-sideways/images/train",
       target_size=(image_size, image_size),
       batch_size=10,
       class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
       directory = "../input/dogs-gone-sideways/images/val",
       target_size=(image_size, image_size),
       class_mode='categorical')

my_new_model.fit_generator(
       train_generator,
       steps_per_epoch=22,
       validation_data=validation_generator,
       validation_steps=1)
data_generator_augumented = ImageDataGenerator(preprocessing_function = preprocess_input, 
                                               horizontal_flip=True, 
                                               width_shift_range = 0.2, 
                                               height_shift_range = 0.2)
train_generator_augumented = data_generator_augumented.flow_from_directory(directory = "../input/dogs-gone-sideways/images/train",
                                                                          target_size = (image_size, image_size), 
                                                                          batch_size = 12,
                                                                          class_mode = "categorical")
validation_generator_augumented = data_generator_augumented.flow_from_directory(directory = "../input/dogs-gone-sideways/images/val",
                                                                               target_size = (image_size, image_size),
                                                                               class_mode = "categorical")
my_new_model.fit_generator(train_generator_augumented, 
                           steps_per_epoch = 19,
                           epochs = 3,                          
                           validation_data = validation_generator_augumented, 
                           validation_steps = 1)