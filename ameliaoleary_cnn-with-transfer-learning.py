# Loading libraries 
from sklearn.datasets import load_files
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend, models, layers, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
# Setting up data 

train_dir = '../input/fruits/fruits-360/Training'
test_dir = '../input/fruits/fruits-360/Test'

# Data augmentation 

batch = 128

train_datagen = ImageDataGenerator(rescale = 1./255, 
                             rotation_range = 20, 
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             zoom_range = 0.2, 
                             horizontal_flip = True, 
                             fill_mode = 'nearest',
                             validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                   target_size = (32, 32),
                                                   batch_size = batch, 
                                                   class_mode = 'categorical', 
                                                   shuffle = True,
                                                   subset = 'training')

validation_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size = (32, 32),
                                                        batch_size = batch, 
                                                        class_mode = 'categorical', 
                                                        shuffle = True,
                                                        subset = 'validation')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                 target_size = (32, 32),
                                                 batch_size = batch,
                                                 class_mode = 'categorical',
                                                 shuffle = False)
# Looking at some visualizations  

import pandas as pd
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

training_data = pd.DataFrame(train_generator.classes, columns=['classes'])
testing_data = pd.DataFrame(validation_generator.classes, columns=['classes'])

def create_stack_bar_data(col, df):
    aggregated = df[col].value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values

x1, y1 = create_stack_bar_data('classes', training_data)
x1 = list(train_generator.class_indices.keys())

trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="Class Count")
layout = dict(height=400, width=1200, title='Class Distribution in Training Data', legend=dict(orientation="h"), 
                yaxis = dict(title = 'Class Count'))
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);
w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
x_batch, y_batch = next(test_generator)
for i in range(1, columns*rows +1):
    image = x_batch[np.random.randint(1,batch-1)]
    fig.add_subplot(rows, columns, i)
    plt.imshow(image)
plt.show()
# Building the model! 

backend.clear_session()

model = Sequential()
model.add(Conv2D(16, kernel_size = 2,input_shape=(32,32,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,kernel_size = 2,activation= 'relu',padding='same'))
model.add(MaxPooling2D(2,2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(120,activation = 'softmax'))

model.summary()

# Compiling the Model 

epoch = 50
trn_cnt = train_generator.n

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit_generator(train_generator, 
                              steps_per_epoch = trn_cnt//(batch*3),
                              epochs = epoch,  
                              validation_data = validation_generator,
                              validation_steps = validation_generator.samples//(batch*3),
                              verbose = 1,
                              shuffle = True,
                              callbacks=[EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True)])

test_loss, test_acc = model.evaluate_generator(test_generator, steps = 50) 
print('test_acc:', test_acc)
# Visualizing loss and accuracy 

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Pre-trained Model 

from tensorflow.keras.applications import VGG16

backend.clear_session()

epoch = 1

conv_base = VGG16 (weights = 'imagenet',   
                  include_top = False,
                  input_shape = (32, 32, 3))

modelVGG = models.Sequential()
modelVGG.add(conv_base) 
modelVGG.add(layers.Flatten())
modelVGG.add(layers.Dense(512, activation = 'relu'))
modelVGG.add(layers.Dense(1, activation = 'sigmoid'))

conv_base.trainable = False

modelVGG.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = modelVGG.fit_generator(
    train_generator,
    steps_per_epoch=200,
    epochs=epoch,
    validation_data=validation_generator,
    validation_steps=50,
    verbose = 2,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True)])


test_loss, test_acc = modelVGG.evaluate_generator(test_generator, steps = 50)

print('VGG_acc:', test_acc)