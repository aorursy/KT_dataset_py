# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Taken from this kernel: https://www.kaggle.com/vipoooool/plant-diseases-classification-using-alexnet
# Docs: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

# Define a data generator with data augmentation transformations for the 
# training dataset.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"

train_set = train_datagen.flow_from_directory(base_dir + '/train',
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              class_mode='categorical')

valid_set = valid_datagen.flow_from_directory(base_dir + '/valid',
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              class_mode='categorical')
base_model = keras.applications.VGG16(
    weights="imagenet",  # load weights pretrained on the ImageNet
    input_shape=(224, 224, 3),
    include_top=False  # do not include the ImageNet classifier at the top
)  
base_model.summary()
# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(224, 224, 3))

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(38)(x)

vgg16_model = keras.Model(inputs, outputs, name='pretrained_vgg16')
vgg16_model.summary()
vgg16_model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.CategoricalAccuracy()]
)

epochs = 5 #10

# train_num = train_set.samples  # num of training samples
# valid_num = valid_set.samples  # num of validation samples

vgg16_history = vgg16_model.fit(train_set,
                                steps_per_epoch=150,  # use 150 random batches (= 4800 samples) for training
                                validation_data=valid_set,
                                epochs=epochs,
                                validation_steps=100,  # use 100 random batches (= 3200 samples) for validation 
)
results = vgg16_model.evaluate(valid_set)
print('val loss:', results[0])
print('val acc:', results[1])
import matplotlib.pyplot as plt
import seaborn as sns

# Set parameters for plotting
plt.rc('figure', figsize=(8, 4))
sns.set(font_scale=1)
train_acc = vgg16_history.history['categorical_accuracy']
val_acc = vgg16_history.history['val_categorical_accuracy']

epochs_list = list(range(1, epochs + 1))

plt.plot(epochs_list, train_acc, label='train acc')
plt.plot(epochs_list, val_acc, label='val acc')
plt.title("VGG-16's Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
train_loss = vgg16_history.history['loss']
val_loss = vgg16_history.history['val_loss']

plt.plot(epochs_list, train_loss, label='train loss')
plt.plot(epochs_list, val_loss, label='val loss')
plt.title("VGG-16's Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
vgg16_model.save('vgg16')
base_model = keras.applications.MobileNet(
    weights="imagenet",  # load weights pretrained on the ImageNet
    input_shape=(224, 224, 3),
    include_top=False  # do not include the ImageNet classifier at the top
)  
base_model.summary()
# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(224, 224, 3))

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(38)(x)

mobilenet_model = keras.Model(inputs, outputs, name='pretrained_mobilenet')
mobilenet_model.summary()
mobilenet_model.compile(optimizer=keras.optimizers.Adam(),
                        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=[keras.metrics.CategoricalAccuracy()]
)

epochs = 5 #10

mobilenet_history = mobilenet_model.fit(train_set,
                                        steps_per_epoch=150,  # use 150 random batches (= 4800 samples) for training
                                        validation_data=valid_set,
                                        epochs=epochs,
                                        validation_steps=100,  # use 100 random batches (= 3200 samples) for validation 
)
results = mobilenet_model.evaluate(valid_set)
print('val loss:', results[0])
print('val acc:', results[1])
train_acc = mobilenet_history.history['categorical_accuracy']
val_acc = mobilenet_history.history['val_categorical_accuracy']

epochs_list = list(range(1, epochs + 1))

plt.plot(epochs_list, train_acc, label='train acc')
plt.plot(epochs_list, val_acc, label='val acc')
plt.title("MobileNet's Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
train_loss = mobilenet_history.history['loss']
val_loss = mobilenet_history.history['val_loss']

plt.plot(epochs_list, train_loss, label='train loss')
plt.plot(epochs_list, val_loss, label='val loss')
plt.title("MobileNet's Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
mobilenet_model.save('mobilenet')
# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
mobilenet_model.summary()
mobilenet_model.compile(optimizer=keras.optimizers.Adam(1e-5),  # set a small learning rate
                        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=[keras.metrics.CategoricalAccuracy()]
)

epochs = 5

mobilenet_ft_history = mobilenet_model.fit(train_set,
                                           steps_per_epoch=150,   # use 150 random batches (= 4800 samples) for training
                                           validation_data=valid_set,
                                           epochs=epochs,
                                           validation_steps=100  # use 100 random batches (= 3200 samples) for validation
)
results = mobilenet_model.evaluate(valid_set)
print('val loss:', results[0])
print('val acc:', results[1])
train_acc = mobilenet_ft_history.history['categorical_accuracy']
val_acc = mobilenet_ft_history.history['val_categorical_accuracy']

epochs_list = list(range(1, epochs + 1))

plt.plot(epochs_list, train_acc, label='train acc')
plt.plot(epochs_list, val_acc, label='val acc')
plt.title("Fine-tuned MobileNet's Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
train_loss = mobilenet_ft_history.history['loss']
val_loss = mobilenet_ft_history.history['val_loss']

plt.plot(epochs_list, train_loss, label='train loss')
plt.plot(epochs_list, val_loss, label='val loss')
plt.title("MobileNet's Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
mobilenet_model.save('mobilenet-fine-tuned')
%%time
# reset the test_data to start iterating over dataset from scratch
valid_set.reset()

# start to predict using vgg-16 model
pred_1 = vgg16_model.predict(valid_set)

# reset the test_data to start iterating over dataset from scratch
valid_set.reset()

# start to predict using mobilenet model
pred_2 = mobilenet_model.predict(valid_set)
# show the vgg-16 prediction result
pred_1
# show the mobilenet prediction result
pred_2
import tensorflow as tf
# use the confusion_matrix function provided by tensorflow to generate confusion matrix of vgg-16
con_mat_1 = tf.math.confusion_matrix(labels=valid_set.classes, predictions=np.argmax(pred_1, axis=1)).numpy()

# normalize the confusion matrix of vgg-16
con_mat_norm_1 = np.around(con_mat_1.astype('float') / con_mat_1.sum(axis=1)[:, np.newaxis], decimals=2)

# convert the nomalized confusion matrix for better view of vgg-16
con_mat_df_1 = pd.DataFrame(con_mat_norm_1,
                     index = valid_set.class_indices.keys(), 
                     columns = valid_set.class_indices.keys())

# use the confusion_matrix function provided by tensorflow to generate confusion matrix of mobilenet
con_mat_2 = tf.math.confusion_matrix(labels=valid_set.classes, predictions=np.argmax(pred_2, axis=1)).numpy()

# normalize the confusion matrix of mobilenet
con_mat_norm_2 = np.around(con_mat_2.astype('float') / con_mat_2.sum(axis=1)[:, np.newaxis], decimals=2)

# convert the nomalized confusion matrix for better view of mobilenet
con_mat_df_2 = pd.DataFrame(con_mat_norm_2,
                     index = valid_set.class_indices.keys(), 
                     columns = valid_set.class_indices.keys())
# show the nomalized confusion matrix from vgg-16
con_mat_df_1
# show the nomalized confusion matrix from mobilenet
con_mat_df_2
# convert the original confusion matrix for better view (using the case numbers) from vgg-16
con_mat_df_explain_1 = pd.DataFrame(con_mat_1,
                     index = valid_set.class_indices.keys(), 
                     columns = valid_set.class_indices.keys())

# convert the original confusion matrix for better view (using the case numbers) from mobilenet
con_mat_df_explain_2 = pd.DataFrame(con_mat_2,
                     index = valid_set.class_indices.keys(), 
                     columns = valid_set.class_indices.keys())
# show the unnomalized confusion matrix from vgg-16
con_mat_df_explain_1
# show the unnomalized confusion matrix from mobilenet
con_mat_df_explain_2
import sklearn.metrics
# generate the clasification report by using the classification_report of sklearn package from vgg-16
report_1 = sklearn.metrics.classification_report(valid_set.classes, np.argmax(pred_1, axis=1), target_names=valid_set.class_indices.keys())
# generate the clasification report by using the classification_report of sklearn package from mobilenet
report_2 = sklearn.metrics.classification_report(valid_set.classes, np.argmax(pred_2, axis=1), target_names=valid_set.class_indices.keys())

# print the report from vgg-16
print(report_1)
# print the report from mobilenet
print(report_2)