from numpy.random import seed
seed(101)

import tensorflow
tensorflow.random.set_seed(101)

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import os
import cv2

import imageio
import skimage
import skimage.io
import skimage.transform

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline





os.mkdir('imagens')



registros = os.listdir('/kaggle/input/breast-histopathology-images/IDC_regular_ps50_idx5')
len(registros)



for registro in registros:
    path_0 = '/kaggle/input/breast-histopathology-images/IDC_regular_ps50_idx5/' + str(registro) + '/0'
    path_1 = '/kaggle/input/breast-histopathology-images/IDC_regular_ps50_idx5/' + str(registro) + '/1'
    
    
    file_list = os.listdir(path_0)
    file_list.extend(os.listdir(path_1))
    
    for filename in file_list:
        # source path to image
        src = os.path.join('/kaggle/input/breast-histopathology-images/IDC_regular_ps50_idx5/' + str(registro) + '/' + filename.split('_')[4][5], filename)
        # destination path to image
        dst = os.path.join('imagens', filename)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
        

len(os.listdir('imagens'))
imgs = os.listdir('imagens')

df = pd.DataFrame(imgs, columns=['filename'])

df['cod'] = df['filename'].apply(lambda x: x.split('_')[0])
df['target'] = df['filename'].apply(lambda x: x.split('_')[4][5])

df.head(10)
df['target'].value_counts()

df_0 = df[df['target'] == '0'].sample(78786, random_state=101)

df_1 = df[df['target'] == '1'].sample(78786, random_state=101)

df_ = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)

df_['target'].value_counts()
y = df_['target']

df_train, df_val = train_test_split(df_, test_size=0.10, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)
'train', df_train['target'].value_counts(),'validation', df_val['target'].value_counts()
train_dir = 'train_dir'
os.mkdir(train_dir)
os.mkdir(train_dir + '/0')
os.mkdir(train_dir + '/1')

val_dir = 'val_dir'
os.mkdir(val_dir)
os.mkdir(val_dir + '/0')
os.mkdir(val_dir + '/1')



train_list = list(df_train['filename'])
val_list = list(df_val['filename'])

for image in train_list:
    # source path to image
    src = os.path.join('imagens', image)
    # destination path to image
    dst = os.path.join(train_dir, image.split('_')[4][5], image)
    # move the image from the source to the destination
    shutil.move(src, dst)


for image in val_list:
    # source path to image
    src = os.path.join('imagens', image)
    # destination path to image
    dst = os.path.join(val_dir, image.split('_')[4][5], image)
    # move the image from the source to the destination
    shutil.move(src, dst)
    

print(len(os.listdir('train_dir/0')))
print(len(os.listdir('train_dir/1')))

print(len(os.listdir('val_dir/0')))
print(len(os.listdir('val_dir/1')))
train_steps = np.ceil(len(train_dir) / 10)
val_steps = np.ceil(len(val_dir) / 10)

datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_dir,
                                        target_size=(50,50),
                                        batch_size=10,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(val_dir,
                                        target_size=(50,50),
                                        batch_size=10,
                                        class_mode='categorical')

test_gen = datagen.flow_from_directory(val_dir,
                                        target_size=(50,50),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# Source: https://www.kaggle.com/fmarazzi/baseline-keras-cnn-roc-fast-5min-0-8253-lb


kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', 
                 input_shape = (50, 50, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()


model.compile(Adam(lr=0.0001), loss='binary_crossentropy', 
              metrics=['accuracy'])


filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=60, verbose=1,
                   callbacks=callbacks_list)

model.metrics_names

# display the loss and accuracy curves

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
history
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# make a prediction
y_pred_keras = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_gen.classes, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
