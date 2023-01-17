import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import glob 


from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D, MaxPool2D
from keras.applications.vgg16 import VGG16,  preprocess_input
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
Dataset_Dir= "../input/covid-19-x-ray-10000-images/dataset"
os.listdir(Dataset_Dir)
normal_images = []
for img_path in glob.glob(Dataset_Dir+"/normal/*"):
    normal_images.append(mpimg.imread(img_path))
    
fig =plt.figure();
fig.suptitle("normal");
plt.imshow(normal_images[0], cmap='gray');


covid_images = []
for img_path in glob.glob(Dataset_Dir+"/covid/*"):
    covid_images.append(mpimg.imread(img_path))
    
fig = plt.figure();
fig.suptitle('covid');
plt.imshow(covid_images[0], cmap='gray');
print("Number of Noraml Images ", len(normal_images))
print("Number of Covid Images ", len(covid_images))
IMG_W = 150
IMG_H = 150
CHANNELS = 3

INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)
NB_CLASSES = 2
EPOCHS = 20
BATCH_SIZE = 6
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=INPUT_SHAPE, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
model.summary()
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  validation_split=0.3)


train_generator = train_datagen.flow_from_directory(
                                    Dataset_Dir,
                                    target_size=(IMG_H, IMG_W),
                                    batch_size=BATCH_SIZE,
                                    class_mode='binary',
                                    subset='training')


validation_generator = train_datagen.flow_from_directory(
                                    Dataset_Dir,
                                    target_size=(IMG_H, IMG_W),
                                    batch_size=BATCH_SIZE,
                                    class_mode='binary',
                                    shuffle=False,
                                    subset='validation')

history = model.fit_generator(
                            train_generator,
                            steps_per_epoch= train_generator.samples //BATCH_SIZE,
                            validation_data=validation_generator,
                            validation_steps=validation_generator.samples//BATCH_SIZE,
                            epochs=EPOCHS)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
label = validation_generator.classes
label
pred = model.predict(validation_generator)
predicted_class_indices = np.where(pred > 0.5, 1, 0)
labels = (validation_generator.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
sample = predicted_class_indices.tolist()
predictions = [j for sub in sample for j in sub]
predictions = [labels2[k] for k in predictions]
print(predicted_class_indices)
print (labels)
print (predictions)
from sklearn.metrics import confusion_matrix

cf = confusion_matrix(predicted_class_indices,label)
cf
plt.matshow(cf)
plt.title('Confusion Matrix Plot')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show();
