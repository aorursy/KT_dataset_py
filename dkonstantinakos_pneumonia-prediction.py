# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import kerastuner as kt
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/train',
                                                  target_size = (64, 64),
                                                  batch_size = 32,
                                                  class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64, 64 , 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
#print(cnn.summary())
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
history = cnn.fit(x = training_set, validation_data = test_set, epochs = 10)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

rec = history.history['recall']
val_rec = history.history['val_recall']

prec = history.history['precision']
val_prec = history.history['val_precision']


epochs_range = range(10)

import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize = (16,8))
plt.subplot(1, 4, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 4, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')

plt.subplot(1, 4, 3)
plt.plot(epochs_range, rec, label = 'Training Recall')
plt.plot(epochs_range, val_rec, label = 'Validation Recall')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Recall')

plt.subplot(1, 4, 4)
plt.plot(epochs_range, prec, label = 'Training Precision')
plt.plot(epochs_range, val_rec, label = 'Validation Precision')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Precision')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = test_image/255.0
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] > 0.5:
    prediction = 'PNEUMONIA'
else:
    prediction = 'NORMAL'
print(prediction)
def cnn_model_builder(hp):
    
    cnn_model = tf.keras.models.Sequential()
    
    cnn_model.add(tf.keras.layers.Conv2D(filters = hp.Int('conv_1_filter', min_value = 32, max_value = 128, step = 32),
                                         kernel_size = hp.Choice('conv_1_kernel', values = [3, 5]),
                                         activation = 'relu',
                                         input_shape = [64, 64 ,3]))
    
    cnn_model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
    
    cnn_model.add(tf.keras.layers.Conv2D(filters = hp.Int('conv_2_filter', min_value = 32, max_value = 128, step = 32),
                                         kernel_size = hp.Choice('conv_2_kernel', values = [3, 5]),
                                         activation = 'relu'))
    
    cnn_model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
    
    
    cnn_model.add(tf.keras.layers.Conv2D(filters = hp.Int('conv_3_filter', min_value = 32, max_value = 128, step = 32),
                                         kernel_size = hp.Choice('conv_3_kernel', values = [3, 5]),
                                         activation = 'relu'))
    
    cnn_model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))    

    cnn_model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.0, 0.5, step = 0.05, default = 0.5)))
    
    cnn_model.add(tf.keras.layers.Flatten())
     
    for z in range(hp.Int('num_layers', 2, 10)):
        cnn_model.add(tf.keras.layers.Dense(units = hp.Int('units_' + str(z), min_value = 32, max_value = 512,step = 32), activation = 'relu'))
    
    
    cnn_model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
    
    cnn_model.compile(optimizer = tf.keras.optimizers.Adam(hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])),
                      loss = 'binary_crossentropy',
                      metrics = ['accuracy'])
    
    return cnn_model
import os
tuner = kt.Hyperband(cnn_model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 50,
                     factor = 3,
                     directory = os.path.normpath('/kaggle/output'))

tuner.search(training_set,
             validation_data = test_set,
             epochs = 30)

# Get the best hyperparameters #
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

# Compile the model with these parameters #
model = tuner.hypermodel.build(best_hps)
history_opt = model.fit(x = training_set, validation_data = test_set, epochs = 10)
