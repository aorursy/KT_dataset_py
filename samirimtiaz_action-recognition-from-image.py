# Importing Essential Libraries
import os
import time
import pandas as pd
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout,LSTM,Conv2D,MaxPool2D,Flatten
# Required Parameters
dataset = "UCF-101/"                                                            # Dataset Path
dataset2 = "dataset/"                                                           # Dataset2 Path
train_path = "/kaggle/input/vid-classification-ucf101/UCF/training_set/"        # Training Path for Kaggle
test_path = "/kaggle/input/vid-classification-ucf101/UCF/testing_set/"          # Testing Path for Kaggle
no_of_frames = 1650                                                             # Number of Frames
ch = 4                                                                          # Model Selection Choice
epochs = 20                                                                     # Number of epochs
batch_size = 10                                                                 # Batch Size
n_classes = 101                                                                 # Number of Classes
patience = 2                                                                    # Patience for EarlyStopping
stime = int(time.time())                                                        # Defining Starting Time



train_datagenerator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagenerator = ImageDataGenerator(rescale=1./255)
model=Sequential()
model.add(Conv2D(112,(3,3),input_shape=(224,224,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='categorical_crossentropy')
train_generator = train_datagenerator.flow_from_directory(train_path,
                                                          target_size = (224, 224),
                                                          color_mode = 'rgb',
                                                          batch_size = batch_size,
                                                          class_mode = 'categorical',
                                                          shuffle = True)


test_generator = test_datagenerator.flow_from_directory(test_path,
                                                        target_size = (224, 224),
                                                        color_mode = 'rgb',
                                                        class_mode = 'categorical')
print(train_generator.class_indices)
print(test_generator.class_indices)
# Training the Model
history=model.fit_generator(
        train_generator,
        steps_per_epoch=151500,
        epochs=4,
        validation_data=test_generator,
        validation_steps=15150
)
# Plotting the Graph
model_history = pd.DataFrame(history.history)
model_history.plot()