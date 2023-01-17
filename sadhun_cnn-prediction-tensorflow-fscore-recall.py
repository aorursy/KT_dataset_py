# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Inspired from https://www.kaggle.com/arjunsarkar/cnn-tensorflow-2-0-f-score-97-recall-98
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import SimpleITK as sitk
import cv2 as cv
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
#class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

#nb_classes = len(class_names)

image_size=150
class_num = class_names.index('mountain')
print(class_num)
def create_training_data(data_dir):              #creating the training data
    
    images = []
    
    for cls in class_names:
        dir = os.path.join(data_dir,cls)
        class_num = class_names.index(cls)
        
        for image in os.listdir(dir):    #going through all the images in different folders and resizing them
            
            image_read = cv.imread(os.path.join(dir,image))
            #we use colors instead of gray scale
            image_read = cv.cvtColor(image_read, cv.COLOR_BGR2RGB)
            image_resized = cv.resize(image_read,(image_size,image_size))
            images.append([image_resized,class_num])
            
    return np.array(images) 
train = create_training_data('/kaggle/input/intel-image-classification/seg_train/seg_train')
test = create_training_data('/kaggle/input/intel-image-classification/seg_test/seg_test')

print(train.shape)
print(test.shape)
plt.imshow(train[2][0], cmap='gray')
print(class_names[train[2][1]])  
#Loading the Images and Labels together

X = []
y = []

for feature, cls in train:
    X.append(feature)          #appending all images
    y.append(cls)            #appending all classes

for feature, cls in test:
    X.append(feature)
    y.append(cls)
print(len(X))
#Reshaping the data for 3 dimension since we are using colors RGB

X_new = np.array(X).reshape(-1, image_size, image_size, 3)
y_new = np.array(y)
del X
del y
print(len(X))
print(X_new.shape)
print(y_new.shape)
y_new = np.expand_dims(y_new, axis =1)
print(y_new.shape)
#Shuffling to mix data
from sklearn.utils import shuffle  
(X_new,y_new)=shuffle(X_new,y_new,random_state=25)
(X_new.shape,y_new.shape)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state = 32)
X_train.shape
X_test.shape
del X_new
del y_new
X_train = X_train / 255            # normalizing
X_test = X_test / 255
print(X_train)
X_train.shape[1:]
#Creating CNN model

i = Input(X_train.shape[1:])                                        # Input Layer

a = Conv2D(32, (3,3), activation ='relu', padding = 'same')(i)      # Convolution
a = BatchNormalization()(a)                                         # Batch Normalization
a = Conv2D(32, (3,3), activation ='relu', padding = 'same')(a)
a = BatchNormalization()(a)
a = MaxPooling2D(2,2)(a)                                            # Max Pooling

a = Conv2D(64, (3,3), activation ='relu', padding = 'same')(a)
a = BatchNormalization()(a)
a = Conv2D(64, (3,3), activation ='relu', padding = 'same')(a)
a = BatchNormalization()(a)
a = MaxPooling2D(2,2)(a)

a = Flatten()(a)                                                      # Flatten
a = Dense(128, activation = 'relu')(a)                               # Fully Connected layer
a = Dropout(0.4)(a)
a = Dense(128, activation = 'relu')(a)
a = Dropout(0.1)(a)

a = Dense(6, activation = 'softmax')(a)                               # Output Layer

model = Model(i,a)
model.compile(optimizer=Adam(lr = 0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()  
batch_size=128
steps_per_epoch=X_train.shape[0]//batch_size
steps_per_epoch
checkpoint = ModelCheckpoint('Pneumonia1.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
train_gen = ImageDataGenerator(rotation_range=10,
                                   horizontal_flip = True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

train_generator = train_gen.flow(X_train,y_train,batch_size)
history = model.fit(train_generator, validation_data=(X_test, y_test), steps_per_epoch = steps_per_epoch, epochs= 15,
                       callbacks = [checkpoint])
#Plotting the losses

plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
pred = model.predict(X_test, batch_size = 8)
pred
pred_final = np.where(pred>0.5,1,0)
pred_final
# Get the confusion matrix
CM = confusion_matrix(y_test, pred_final)

fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(8,8))
plt.title('Confusion matrix')
plt.xticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.yticks(range(2), ['Normal','Pneumonia'], fontsize=10)
plt.show()
def perf_measure(y_test, pred_final):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(pred_final)): 
        if y_test[i]==pred_final[i]==1:
           TP += 1
        if y_test[i]==1 and y_test[i]!=pred_final[i]:
           FP += 1
        if y_test[i]==pred_final[i]==0:
           TN += 1
        if y_test[i]==0 and y_test[i]!=pred_final[i]:
           FN += 1

    return(TP, FP, TN, FN)


tp, fp, tn ,fn = perf_measure(y_test,pred_final)

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f_score = (2*precision*recall)/(precision+recall)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
print("F-Score is {:.2f}".format(f_score))