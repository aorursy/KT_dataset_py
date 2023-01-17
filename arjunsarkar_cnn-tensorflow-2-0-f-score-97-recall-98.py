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
image_size = 256       #resize all images to 256*256

labels = ['NORMAL', 'PNEUMONIA']          #labels from the folders
def create_training_data(data_dir):              #creating the training data
    
    images = []
    
    for label in labels:
        dir = os.path.join(data_dir,label)
        class_num = labels.index(label)
        
        for image in os.listdir(dir):    #going through all the images in different folders and resizing them
            
            image_read = cv.imread(os.path.join(dir,image),cv.IMREAD_GRAYSCALE)
            image_resized = cv.resize(image_read,(image_size,image_size))
            images.append([image_resized,class_num])
            
    return np.array(images)
train = create_training_data('/kaggle/input/chest-xray-pneumonia/chest_xray/train')
test = create_training_data('/kaggle/input/chest-xray-pneumonia/chest_xray/test')
val = create_training_data('/kaggle/input/chest-xray-pneumonia/chest_xray/val')
train.shape
test.shape
val.shape
plt.imshow(train[1][0], cmap='gray')
print(labels[train[1][1]])   
plt.imshow(train[5000][0], cmap='gray')
print(labels[train[5000][1]])
X = []
y = []

for feature, label in train:
    X.append(feature)          #appending all images
    y.append(label)            #appending all labels

for feature, label in test:
    X.append(feature)
    y.append(label)
X_new = np.array(X).reshape(-1, image_size, image_size, 1)
y_new = np.array(y)
y_new = np.expand_dims(y_new, axis =1)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state = 32)
X_train.shape
y_train.shape
X_train = X_train / 255            # normalizing
X_test = X_test / 255
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

a = Conv2D(128, (3,3), activation ='relu', padding = 'same')(a)
a = BatchNormalization()(a)
a = Conv2D(128, (3,3), activation ='relu', padding = 'same')(a)
a = BatchNormalization()(a)
a = MaxPooling2D(2,2)(a)

a = Conv2D(256, (3,3), activation ='relu', padding = 'same')(a)
a = BatchNormalization()(a)
a = Conv2D(256, (3,3), activation ='relu', padding = 'same')(a)
a = BatchNormalization()(a)
a = MaxPooling2D(2,2)(a)

a = Flatten()(a)                                                      # Flatten
a = Dense(512, activation = 'relu')(a)                               # Fully Connected layer
a = Dropout(0.4)(a)
a = Dense(512, activation = 'relu')(a)
a = Dropout(0.3)(a)
a = Dense(512, activation = 'relu')(a)
a = Dropout(0.1)(a)

a = Dense(1, activation = 'sigmoid')(a)                               # Output Layer

model = Model(i,a)
model.compile(optimizer=Adam(lr = 0.0001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()    
batch_size = 4

train_gen = ImageDataGenerator(rotation_range=10,
                                   horizontal_flip = True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

train_generator = train_gen.flow(X_train,y_train,batch_size)
steps_per_epoch = X_train.shape[0]//batch_size
checkpoint = ModelCheckpoint('Pneumonia1.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

#to save the model - epochs with best validation loss
r = model.fit(train_generator, validation_data=(X_test, y_test), steps_per_epoch = steps_per_epoch, epochs= 15,
                       callbacks = [checkpoint])
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
new_model = tf.keras.models.load_model('Pneumonia1.h5')   #loading model to train further
new_model.compile(optimizer = Adam(lr = 0.00001), loss = 'binary_crossentropy', metrics = ['accuracy']) 

checkpoint1 = ModelCheckpoint('Pneumonia2.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

batch_size = 4

r1 = new_model.fit(train_generator, validation_data=(X_test, y_test), steps_per_epoch = steps_per_epoch, epochs= 10,
                       callbacks = [checkpoint1])
final_model = tf.keras.models.load_model('Pneumonia2.h5')
pred = final_model.predict(X_test, batch_size = 8)
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