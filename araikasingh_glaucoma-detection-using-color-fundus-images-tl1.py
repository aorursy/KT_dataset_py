# Version 2
# Preprocessing: Optical Disk localised manually
# --Very few data to create an automated optical disk optimisation
# Transfer learning on already trained ResNet50. Compile process.

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

categories = 2

model = Sequential()

# The fully connected top layer of ResNet50 is not to added in this model
model.add(ResNet50(include_top = False, pooling = 'avg', weights = 'imagenet'))

# All inputs and outputs are connected to neurons (Dense Layers)
# ReLu activation can be used here. Difference --?
model.add(Dense(categories, activation='softmax'))
model.layers[0].trainable = False

model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
import os
print(os.listdir("../input/drishtiiiitlocalisedfundus/Test/Test"))
import glob
pngCounter = len(glob.glob1('../input/drishtiiiitlocalisedfundus/Test/Test/Glaucomatous',"*.png")) + len(glob.glob1('../input/drishtiiiitlocalisedfundus/Test/Test/Normal',"*.png"))
print(pngCounter)
import glob
pngCounter = len(glob.glob1('../input/drishtiiiitlocalisedfundus/Training/Training/Glaucomatous',"*.png")) + len(glob.glob1('../input/drishtiiiitlocalisedfundus/Training/Training/Normal',"*.png"))
print(pngCounter)
model.summary()
#Model Fitting

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

data_generator = ImageDataGenerator(preprocess_input)

image_size = 224

train_generator = data_generator.flow_from_directory(directory = '../input/drishtiiiitlocalisedfundus/Training/Training',
                                                     target_size = (image_size, image_size),
                                                     batch_size = 5,
                                                     class_mode = 'categorical')

validation_generator = data_generator.flow_from_directory(directory = '../input/drishtiiiitlocalisedfundus/Test/Test',
                                                          target_size = (image_size, image_size),
                                                         batch_size = 51,
                                                         class_mode = 'categorical')

model_fitting = model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        epochs = 3,
        validation_data=validation_generator,
        validation_steps=1
)