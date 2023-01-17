import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import shutil   
from keras import backend as K
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History 
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
train = '/kaggle/input/10-monkey-species/training/training/'
val = '/kaggle/input/10-monkey-species/validation/validation/'
labels = pd.read_csv("/kaggle/input/10-monkey-species/monkey_labels.txt")
labels
# Total number of training images
num_of_train_samples = 0
for train_dataset in os.listdir(train):
    in_folder = train + "/" + train_dataset 
    in_folder_list = os.listdir(in_folder)
    num_of_train_samples = num_of_train_samples + len(in_folder_list)
print("Number of Training samples   : ",num_of_train_samples)

# Total number of validation images
num_of_validation_samples = 0
for validation_dataset in os.listdir(val):
    in_folder_val = val + "/" + validation_dataset
    in_folder_val_list = os.listdir(in_folder_val)
    num_of_validation_samples = num_of_validation_samples + len(in_folder_val_list)
print("Number of Validation samples : ", num_of_validation_samples)
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 16
learning_rate = 0.0001
epoch = 35

# Defining image width and height respectively
img_rows = 256
img_cols = 256
train_generator = train_datagen.flow_from_directory(train,
                                                    target_size = (img_rows, img_cols),
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(val,
                                                        target_size = (img_rows, img_cols),
                                                        batch_size = batch_size,
                                                        shuffle = False, class_mode='categorical')
steps_per_epoch = num_of_train_samples // batch_size
print("Steps per epoch: ",steps_per_epoch)
from keras.applications import ResNet50
# The sequential API allows you to create models layer-by-layer
resnet_model = Sequential()
resnet_model.add(ResNet50(include_top=False, 
                   pooling='max', 
                   weights='imagenet'))
resnet_model.add(Dense(10, activation="softmax"))

# Summary: to find the number of parameters
resnet_model.layers[0].trainable=False
resnet_model.summary()

sgd = optimizers.SGD(lr=learning_rate, decay=0.00001,momentum = 0.0,nesterov=False)
resnet_model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])
# Trains the model for a given number of epochs (iterations on a dataset).
resnet_training = resnet_model.fit_generator(train_generator,
                               steps_per_epoch = steps_per_epoch,
                               epochs = epoch,
                               validation_data = validation_generator,
                               validation_steps = num_of_validation_samples // batch_size)
training_accuracy_resnet      = resnet_training.history['accuracy'][-1]
training_loss_resnet          = resnet_training.history['loss'][-1]
validation_accuracy_resnet    = resnet_training.history['val_accuracy'][-1]
validation_loss_resnet        = resnet_training.history['val_loss'][-1]
print("Training Accuracy ResNet   :", training_accuracy_resnet )
print("Training Loss ResNet       :", training_loss_resnet)
print("Validation Accuracy ResNet :", validation_accuracy_resnet)
print("Validation Loss ResNet     :", validation_loss_resnet)
# Generating Confusion Matrix and Classification Report
Y_pred_res = resnet_model.predict_generator(validation_generator, num_of_validation_samples // batch_size+1)
y_pred_res = np.argmax(Y_pred_res, axis=1)
print('Confusion Matrix')
conf_matrix_res = confusion_matrix(validation_generator.classes, y_pred_res)
cm_res = np.array2string(conf_matrix_res)
print(conf_matrix_res)
print("=============================================================================================")
print('Classification Report')
target_names = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9']
class_rep_res = classification_report(validation_generator.classes, y_pred_res, target_names=target_names)
print(class_rep_res)

epoch_inc = 30
learning_rate_inc = 0.001
batch_size_inc = 32
steps_per_epoch_inc = num_of_train_samples // batch_size_inc
print("Steps per epoch: ",steps_per_epoch_inc)
from keras.applications import InceptionV3
# The sequential API allows us to create model layer by layer
inc_model = Sequential()
inc_model.add(InceptionV3(include_top=False, 
                      pooling='max',
                      weights='imagenet'))
inc_model.add(Dense(10, activation="softmax"))

# Summary: to find the number of parameters
inc_model.layers[0].trainable=False
inc_model.summary()

adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
inc_model.compile(loss="categorical_crossentropy",
              optimizer=adam,
              metrics=["accuracy"])

# Trains the model for a given number of epochs (iterations on a dataset).
inc_training = inc_model.fit_generator(train_generator,
                                       steps_per_epoch = steps_per_epoch_inc,
                                       epochs = epoch_inc,
                                       validation_data = validation_generator,
                                       validation_steps = num_of_validation_samples // batch_size_inc)
training_accuracy_inc      = inc_training.history['accuracy'][-1]
training_loss_inc          = inc_training.history['loss'][-1]
validation_accuracy_inc    = inc_training.history['val_accuracy'][-1]
validation_loss_inc        = inc_training.history['val_loss'][-1]
print("Training Accuracy Inception   :", training_accuracy_inc )
print("Training Loss Inception       :", training_loss_inc)
print("Validation Accuracy Inception :", validation_accuracy_inc)
print("Validation Loss Inception     :", validation_loss_inc)
# Generating Confusion Matrix and Classification Report
Y_pred_inc = inc_model.predict_generator(validation_generator, num_of_validation_samples // batch_size+1)
y_pred_inc = np.argmax(Y_pred_inc, axis=1)
print('Confusion Matrix')
conf_matrix_inc = confusion_matrix(validation_generator.classes, y_pred_inc)
cm_inc = np.array2string(conf_matrix_inc)
print(conf_matrix_inc)
print("=============================================================================================")
print('Classification Report')
target_names = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9']
class_rep_inc = classification_report(validation_generator.classes, y_pred_inc, target_names=target_names)
print(class_rep_inc)
epoch_vgg = 30
learning_rate_vgg = 0.001
batch_size_vgg = 64
steps_per_epoch_vgg = num_of_train_samples // batch_size_vgg
print("Steps per epoch: ",steps_per_epoch_vgg)
from keras.applications import vgg16
    # The sequential API allows you to create models layer-by-layer
vgg_model=Sequential()
vgg_model.add(vgg16.VGG16(include_top = False, pooling = 'max', weights = 'imagenet'))
vgg_model.add(Dense(10, activation="softmax"))

    # Summary: to find the number of parameters
vgg_model.layers[0].trainable=False
vgg_model.summary() 

adam = optimizers.Adam(lr=learning_rate_vgg, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
vgg_model.compile(loss="categorical_crossentropy",
                  optimizer=adam,
                  metrics=["accuracy"])
# Trains the model for a given number of epochs (iterations on a dataset).
vgg_training = vgg_model.fit_generator(train_generator,
                                       steps_per_epoch = steps_per_epoch_vgg,
                                       epochs = epoch_vgg,
                                       validation_data = validation_generator,
                                       validation_steps = num_of_validation_samples // batch_size_vgg)
training_accuracy_vgg      = vgg_training.history['accuracy'][-1]
training_loss_vgg          = vgg_training.history['loss'][-1]
validation_accuracy_vgg    = vgg_training.history['val_accuracy'][-1]
validation_loss_vgg        = vgg_training.history['val_loss'][-1]
print("Training Accuracy VGG    :", training_accuracy_vgg )
print("Training Loss VGG        :", training_loss_vgg)
print("Validation Accuracy VGG  :", validation_accuracy_vgg)
print("Validation Loss VGG      :", validation_loss_vgg)
# Generating Confusion Matrix and Classification Report
Y_pred_vgg = vgg_model.predict_generator(validation_generator, num_of_validation_samples // batch_size+1)
y_pred_vgg = np.argmax(Y_pred_vgg, axis=1)
print('Confusion Matrix')
conf_matrix_vgg = confusion_matrix(validation_generator.classes, y_pred_vgg)
cm_vgg = np.array2string(conf_matrix_vgg)
print(conf_matrix_vgg)
print("=============================================================================================")
print('Classification Report')
target_names = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9']
class_rep_vgg = classification_report(validation_generator.classes, y_pred_vgg, target_names=target_names)
print(class_rep_vgg)
epoch_cn = 30
learning_rate_cn = 0.001
batch_size_cn = 64
steps_per_epoch_cn = num_of_train_samples // batch_size_cn
print("Steps per epoch: ",steps_per_epoch_cn)
# Custom network
model_cn = Sequential()
model_cn.add(Conv2D(16,(3,3),input_shape=(256,256,3),padding='same'))
model_cn.add(Activation('relu'))
model_cn.add(BatchNormalization())
model_cn.add(MaxPooling2D(pool_size=(2,2)))

model_cn.add(Conv2D(32,(3,3),padding='same'))
model_cn.add(Activation('relu'))
model_cn.add(BatchNormalization())
model_cn.add(MaxPooling2D(pool_size=(2,2)))
model_cn.add(Dropout(0.25))


model_cn.add(Conv2D(32,(3,3),padding='same'))
model_cn.add(Activation('relu'))
model_cn.add(BatchNormalization())
model_cn.add(MaxPooling2D(pool_size=(2,2)))

model_cn.add(Conv2D(64,(3,3),padding='same'))
model_cn.add(Activation('relu'))
model_cn.add(BatchNormalization())
model_cn.add(MaxPooling2D(pool_size=(2,2)))

model_cn.add(Flatten())
model_cn.add(Dense(256,activation='relu'))
#model.add(LeakyReLU(0.1))
model_cn.add(Dropout(0.5))
model_cn.add(Dense(10))
model_cn.add(Activation("softmax"))

model_cn.summary()


model_cn.compile(loss="categorical_crossentropy",
                  optimizer= 'adam',
                  metrics=["accuracy"])
# Trains the model for a given number of epochs (iterations on a dataset).
cn_training = model_cn.fit_generator(train_generator,
                                       steps_per_epoch = steps_per_epoch_cn,
                                       epochs = epoch_cn,
                                       validation_data = validation_generator,
                                       validation_steps = num_of_validation_samples // batch_size_cn)
training_accuracy_cn      = cn_training.history['accuracy'][-1]
training_loss_cn          = cn_training.history['loss'][-1]
validation_accuracy_cn    = cn_training.history['val_accuracy'][-1]
validation_loss_cn        = cn_training.history['val_loss'][-1]
print("Training Accuracy CN    :", training_accuracy_cn )
print("Training Loss CN        :", training_loss_cn)
print("Validation Accuracy CN  :", validation_accuracy_cn)
print("Validation Loss CN      :", validation_loss_cn)
# Generating Confusion Matrix and Classification Report
Y_pred_cn = model_cn.predict_generator(validation_generator, num_of_validation_samples // batch_size+1)
y_pred_cn = np.argmax(Y_pred_cn, axis=1)
print('Confusion Matrix')
conf_matrix_cn = confusion_matrix(validation_generator.classes, y_pred_cn)
cm_cn = np.array2string(conf_matrix_cn)
print(conf_matrix_cn)
print("=============================================================================================")
print('Classification Report')
# target_names = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9']
class_rep_cn = classification_report(validation_generator.classes, y_pred_cn, target_names=target_names)
print(class_rep_cn)

model_comp = pd.DataFrame({"Models": ['ResNet50', 'Inception', 'VGG16', 'Custom Network'],
                           "Batch Size":[16,32,64,64],
                           "Epochs":[45,30,30,30],
                           "Learning Rate": [0.0001,0.001,0.001,0.001],
                           "Steps per epoch":[68,34,17,17],
                           "Image Resolution":['256*256','256*256', '256*256', '256*256'],
                          "Training Accuracy": [training_accuracy_resnet,training_accuracy_inc,training_accuracy_vgg,training_accuracy_cn],
                          "Training Loss": [training_loss_resnet,training_loss_inc,training_loss_vgg,training_loss_cn],
                          "Validation Accuracy": [validation_accuracy_resnet,validation_accuracy_inc,validation_accuracy_vgg,validation_accuracy_cn],
                          "Validation Loss": [validation_loss_resnet,validation_loss_inc,validation_loss_vgg,validation_loss_cn],
                          })
model_comp