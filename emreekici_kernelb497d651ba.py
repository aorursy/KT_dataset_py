#!pip install tf-nightly
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_name = 'dataset'
image_size = (180, 180)
batch_size = 64
epochs = 30

# Create a train dataset
data_generator = ImageDataGenerator(rescale=1./255,
                                    #shear_range=0.1,
                                    #zoom_range=0.2,
                                    #width_shift_range=0.1,
                                    #height_shift_range=0.1,
                                    validation_split=0.2)

train_generator = data_generator.flow_from_directory("../input/my-dataset/dataset",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    target_size=image_size,
                                                    shuffle=True, 
                                                    subset="training")

validation_generator = data_generator.flow_from_directory("../input/my-dataset/dataset",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    target_size=image_size,
                                                    shuffle=True,  
                                                    subset="validation")
from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(180, 180, 3))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=["accuracy"])

hist = model.fit(
    train_generator, validation_data=validation_generator, batch_size=batch_size, epochs=epochs,
)
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))
