# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
from numpy.random import seed
seed(45)

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline
train_dirname = '/kaggle/input/histopathologic-cancer-detection/train'
train_labels = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
train_labels.head()
train_labels['label'].value_counts()
positive_samples = train_labels.loc[train_labels['label'] == 1].sample(4)
negative_samples = train_labels.loc[train_labels['label'] == 0].sample(4)
positive_images = []
negative_images = []
for sample in positive_samples['id']:
    path = os.path.join(train_dirname, sample+'.tif')
    img = cv.imread(path)
    positive_images.append(img)
        
for sample in negative_samples['id']:
    path = os.path.join(train_dirname, sample+'.tif')
    img = cv.imread(path)
    negative_images.append(img)

fig,axis = plt.subplots(2,4,figsize=(20,8))
fig.suptitle('Dataset samples presentation plot',fontsize=20)
for i,img in enumerate(positive_images):
    axis[0,i].imshow(img)
    rect = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='g',facecolor='none', linestyle=':', capstyle='round')
    axis[0,i].add_patch(rect)
axis[0,0].set_ylabel('Positive samples', size='large')
for i,img in enumerate(negative_images):
    axis[1,i].imshow(img)
    rect = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',facecolor='none', linestyle=':', capstyle='round')
    axis[1,i].add_patch(rect)
axis[1,0].set_ylabel('Negative samples', size='large')
    
preprocessed_dir = 'train_for_tf'
os.mkdir(preprocessed_dir)

validation_dir = 'val_dir'
training_dir = 'train_dir'
test_dir = 'test_dir'
positive_label_dir = 'positive'
negative_label_dir = 'negative'
os.mkdir(os.path.join(preprocessed_dir,validation_dir))
os.mkdir(os.path.join(preprocessed_dir,training_dir))
os.mkdir(os.path.join(preprocessed_dir,test_dir))
os.mkdir(os.path.join(preprocessed_dir,validation_dir,positive_label_dir))
os.mkdir(os.path.join(preprocessed_dir,validation_dir,negative_label_dir))
os.mkdir(os.path.join(preprocessed_dir,training_dir,positive_label_dir))
os.mkdir(os.path.join(preprocessed_dir,training_dir,negative_label_dir))
os.mkdir(os.path.join(preprocessed_dir,test_dir,positive_label_dir))
os.mkdir(os.path.join(preprocessed_dir,test_dir,negative_label_dir))
IMG_SIZE = 96
IMG_CHANNELS = 3
TRAIN_SIZE=80000
# TRAIN_SIZE=89000
BATCH_SIZE = 16
EPOCHS = 30
train_neg = train_labels[train_labels['label']==0].sample(TRAIN_SIZE,random_state=45)
train_pos = train_labels[train_labels['label']==1].sample(TRAIN_SIZE,random_state=45)

train_data = pd.concat([train_neg, train_pos], axis=0).reset_index(drop=True)

train_data = shuffle(train_data)
train_data['label'].value_counts()
train_data.head()
y = train_data['label']
train_df, val_df = train_test_split(train_data, test_size=0.3, random_state=45, stratify=y)
y = val_df['label']
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=45, stratify=y)
print(train_df.shape)
print(val_df.shape)
print(test_df.shape)
for sample in train_df.iterrows():
    source = os.path.join(train_dirname, sample[1]['id']+'.tif')
    if sample[1]['label'] == 0:
        label = 'positive'
    else:
        label = 'negative'
    target = os.path.join(preprocessed_dir,training_dir,label,sample[1]['id']+'.tif')
    shutil.copyfile(source, target)
    
for sample in val_df.iterrows():
    source = os.path.join(train_dirname, sample[1]['id']+'.tif')
    if sample[1]['label'] == 0:
        label = 'positive'
    else:
        label = 'negative'
    target = os.path.join(preprocessed_dir,validation_dir,label,sample[1]['id']+'.tif')
    shutil.copyfile(source, target)

for sample in test_df.iterrows():
    source = os.path.join(train_dirname, sample[1]['id']+'.tif')
    if sample[1]['label'] == 0:
        label = 'positive'
    else:
        label = 'negative'
    target = os.path.join(preprocessed_dir,test_dir,label,sample[1]['id']+'.tif')
    shutil.copyfile(source, target)
print(len(os.listdir('train_for_tf/train_dir/positive')))
print(len(os.listdir('train_for_tf/train_dir/negative')))
print(len(os.listdir('train_for_tf/val_dir/positive')))
print(len(os.listdir('train_for_tf/val_dir/negative')))
print(len(os.listdir('train_for_tf/test_dir/positive')))
print(len(os.listdir('train_for_tf/test_dir/negative')))
TRAIN_PATH = 'train_for_tf/train_dir'
VAL_PATH = 'train_for_tf/val_dir'
TEST_PATH = 'train_for_tf/test_dir'
total_train = len(os.listdir('train_for_tf/train_dir/positive')) + len(os.listdir('train_for_tf/train_dir/negative'))
total_val = len(os.listdir('train_for_tf/val_dir/positive')) + len(os.listdir('train_for_tf/val_dir/negative'))
total_test = len(os.listdir('train_for_tf/test_dir/positive')) + len(os.listdir('train_for_tf/test_dir/negative'))
train_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=TRAIN_PATH,
                                                           shuffle=True,
                                                           target_size=(IMG_SIZE, IMG_SIZE),
                                                           class_mode='binary')

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=VAL_PATH,
                                                              target_size=(IMG_SIZE,IMG_SIZE),
                                                              class_mode='binary')

test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
test_data_gen = test_image_generator.flow_from_directory(batch_size=1,
                                                              directory=TEST_PATH,
                                                              target_size=(IMG_SIZE,IMG_SIZE),
                                                              class_mode='binary',shuffle=False)
simple_model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE ,3)),
    MaxPooling2D(),
    Dropout(0.3),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.3),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

simple_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

simple_model.summary()
EPOCHS = 25
checkpoint_filepath = 'checkpoint_simple_model.hdf5'
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
early_stop = EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)                             
callbacks_list = [checkpoint, reduce_lr, early_stop]

simple_history = simple_model.fit_generator(train_data_gen, steps_per_epoch=total_train//BATCH_SIZE, 
                    validation_data=val_data_gen,
                    validation_steps=total_val//BATCH_SIZE,
                    epochs=EPOCHS, verbose=1,
                   callbacks=callbacks_list)
resnet50_train_image_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
resnet50_train_data_gen = resnet50_train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=TRAIN_PATH,
                                                           shuffle=True,
                                                           target_size=(IMG_SIZE, IMG_SIZE),
                                                           class_mode='binary')
resnet50_validation_image_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
resnet50_val_data_gen = resnet50_validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=VAL_PATH,
                                                              target_size=(IMG_SIZE,IMG_SIZE),
                                                              class_mode='binary')
resnet50_test_image_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
resnet50_test_data_gen = resnet50_test_image_generator.flow_from_directory(batch_size=1,
                                                              directory=TEST_PATH,
                                                              target_size=(IMG_SIZE,IMG_SIZE),
                                                            class_mode='binary',shuffle=False)
from tensorflow.keras.applications.resnet50 import ResNet50

dropout_fc = 0.3

resnet50_base_model = ResNet50(weights = 'imagenet', include_top = False,pooling = max, input_shape = (IMG_SIZE,IMG_SIZE,3))
resnet50_model = Sequential([
    resnet50_base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(dropout_fc),
    Dense(1,activation="sigmoid")
])
resnet50_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

resnet50_model.summary()
EPOCHS = 60
# needs generator that zero centers the data, without rescaling
resnet50_filepath = "checkpoint_resnet50_model.h5"
checkpoint = ModelCheckpoint(resnet50_filepath, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
early_stop = EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)                             
callbacks_list = [checkpoint, reduce_lr, early_stop]

resnet50_history = resnet50_model.fit_generator(resnet50_train_data_gen, steps_per_epoch=total_train//BATCH_SIZE, 
                    validation_data=resnet50_val_data_gen,
                    validation_steps=total_val//BATCH_SIZE,
                    epochs=EPOCHS, verbose=1,
                   callbacks=callbacks_list)
mobilenetv2_train_image_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
mobilenetv2_train_data_gen = mobilenetv2_train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=TRAIN_PATH,
                                                           shuffle=True,
                                                           target_size=(IMG_SIZE, IMG_SIZE),
                                                           class_mode='binary')
mobilenetv2_validation_image_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) # Generator for our validation data
mobilenetv2_val_data_gen = mobilenetv2_validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=VAL_PATH,
                                                              target_size=(IMG_SIZE,IMG_SIZE),
                                                              class_mode='binary')
mobilenetv2_test_image_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input) # Generator for our validation data
mobilenetv2_test_data_gen = mobilenetv2_test_image_generator.flow_from_directory(batch_size=1,
                                                              directory=TEST_PATH,
                                                              target_size=(IMG_SIZE,IMG_SIZE),
                                                              class_mode='binary',shuffle=False)
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

dropout_fc = 0.3
mobilenetv2_base_model = MobileNetV2(weights = 'imagenet', include_top = False, pooling = max, input_shape = (IMG_SIZE,IMG_SIZE,3))
mobilenetv2_model = Sequential([
    mobilenetv2_base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(dropout_fc),
    Dense(1,activation="sigmoid")
])
mobilenetv2_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

mobilenetv2_model.summary()
EPOCHS = 50
mobilenetv2_filepath = "mobilenetv2_model.h5"
checkpoint = ModelCheckpoint(mobilenetv2_filepath, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
early_stop = EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)                             
callbacks_list = [checkpoint, reduce_lr, early_stop]

mobilenetv2_history = mobilenetv2_model.fit_generator(mobilenetv2_train_data_gen, steps_per_epoch=total_train//BATCH_SIZE, 
                    validation_data=mobilenetv2_val_data_gen,
                    validation_steps=total_val//BATCH_SIZE,
                    epochs=EPOCHS, verbose=1,
                   callbacks=callbacks_list)
plt.figure(figsize=(20,5))
plt.subplot(221)
plt.plot(resnet50_history.history['loss'], color='orange', label="training_loss")
plt.plot(resnet50_history.history['val_loss'], color='blue', label="validation_loss")
plt.legend(loc='best')
plt.title('training plot -  - ResNet50')
plt.xlabel('epoch')
plt.savefig("training.png", bbox_inches='tight')

plt.subplot(222)
plt.plot(resnet50_history.history['accuracy'], color='orange', label="training_accuracy")
plt.plot(resnet50_history.history['val_accuracy'], color='blue',label="validation_accuracy")
plt.legend(loc='best')
plt.title('validation plot - ResNet50')
plt.xlabel('epoch')
plt.savefig("validation.png", bbox_inches='tight')
plt.show()
plt.figure(figsize=(20,5))
plt.subplot(223)
plt.plot(mobilenetv2_history.history['loss'], color='orange', label="training_loss")
plt.plot(mobilenetv2_history.history['val_loss'], color='blue', label="validation_loss")
plt.legend(loc='best')
plt.title('training plot -  - MobileNetV2')
plt.xlabel('epoch')
plt.savefig("training.png", bbox_inches='tight')

plt.subplot(224)
plt.plot(mobilenetv2_history.history['accuracy'], color='orange', label="training_accuracy")
plt.plot(mobilenetv2_history.history['val_accuracy'], color='blue',label="validation_accuracy")
plt.legend(loc='best')
plt.title('validation plot - MobileNetV2')
plt.xlabel('epoch')
plt.savefig("validation.png", bbox_inches='tight')
plt.show()
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# make a prediction resnet50
resnet50_model_predictions = resnet50_model.predict_generator(resnet50_test_data_gen, steps=total_test, verbose=1)
fpr_resnet50, tpr_resnet50, thresholds_resnet50 = roc_curve(resnet50_test_data_gen.classes, resnet50_model_predictions)
resnet50_model_auc = roc_auc_score(resnet50_test_data_gen.classes, resnet50_model_predictions)
print(f'ResNet50 AUC = {resnet50_model_auc}')

# make a prediction MobileNetV2
mobilenetv2_model_predictions = mobilenetv2_model.predict_generator(mobilenetv2_test_data_gen, steps=total_test, verbose=1)
fpr_mobilenetv2, tpr_mobilenetv2, thresholds_mobilenetv2 = roc_curve(mobilenetv2_test_data_gen.classes, mobilenetv2_model_predictions)
mobilenetv2_model_auc = roc_auc_score(mobilenetv2_test_data_gen.classes, mobilenetv2_model_predictions)
print(f'MobileNetV2 AUC = {mobilenetv2_model_auc}')
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_resnet50, tpr_resnet50, label='area = {:.4f}'.format(resnet50_model_auc))
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ResNet50 ROC curve')
plt.legend(loc='best')
plt.show()
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_mobilenetv2, tpr_mobilenetv2, label='area = {:.4f}'.format(mobilenetv2_model_auc))
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('MobileNetV2 ROC curve')
plt.legend(loc='best')
plt.show()