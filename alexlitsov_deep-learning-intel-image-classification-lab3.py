from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import layers

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.optimizers import RMSprop

from sklearn.utils import shuffle

from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model



import numpy as np



from IPython.display import SVG



from datetime import datetime
BATCH = 128

datagen = ImageDataGenerator(rescale=1./255)





    

train_data = datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_train/seg_train',

                                        target_size=(150, 150),

                                        batch_size=BATCH,

                                        class_mode='categorical',

                                        shuffle=True)



test_data = datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_test/seg_test',

                                        target_size=(150, 150),

                                        batch_size=BATCH,

                                        class_mode='categorical',

                                        shuffle=True)
train_data.class_indices
EPOCHS = 15
import subprocess

import pprint



sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)



out_str = sp.communicate()

out_list = str(out_str[0]).split('\\n')



out_dict = {}



for item in out_list:

    print(item)
def saveModel(model, filename):

    model.summary()

    plot_model(model,to_file=filename,show_shapes=True, expand_nested=True)
def addDenseLayers(model):

    denseLayers = [

        layers.Dense(1024, activation='relu'),

        layers.Dense(256, activation='relu'),

        layers.Dense(64, activation='relu'),

        layers.Dense(6, activation='softmax')

    ]

    for layer in denseLayers:

        model.add(layer)
model = Sequential([

    layers.Conv2D(3, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.Flatten(),

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model1.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(3, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model1_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(3, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model2.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(3, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model2_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(6, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model3.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(6, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model3_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(6, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model4.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(6, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model4_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(12, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model5.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(12, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model5_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(12, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model6.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(12, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model6_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(6, (3, 3), activation = 'relu', padding = 'same'),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model7.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(6, (3, 3), activation = 'relu', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model7_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(32, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(6, (3, 3), activation = 'tanh', padding = 'same'),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model8.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(32, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(6, (3, 3), activation = 'tanh', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model8_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model9.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model9_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(32, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation = 'tanh', padding = 'same'),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model10.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(32, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation = 'tanh', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model10_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model11.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(128, (3, 3), activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model11_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(128, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation = 'tanh', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation = 'tanh', padding = 'same'),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model12.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)
model = Sequential([

    layers.Conv2D(128, (3, 3), activation = 'tanh', padding = 'same', input_shape = (150, 150, 3)),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation = 'tanh', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation = 'tanh', padding = 'same'),

    layers.MaxPooling2D(2, 2),

    layers.Flatten()

])

addDenseLayers(model)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])



saveModel(model, 'model12_pooling.png')



time_start = datetime.now()

history = model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

print('Time: ', datetime.now() - time_start)