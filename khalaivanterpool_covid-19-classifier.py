#!/usr/bin/env python
# coding: utf-8

# In[64]:


#makes keras backend
from keras import backend as K


# In[65]:


#generate image/load img as PIL/convert PIL to numpy array
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array


# In[66]:


#imports model and layer grouping/feature
from keras.models import Sequential, Model


# In[67]:


#layers for models
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D, MaxPool2D


# In[68]:


#CNN with 16 layers, 138Mparameters
from keras.applications.vgg16 import VGG16, preprocess_input


# In[69]:


#optimizers for Hyperparameters --learning rate
from keras.optimizers import Adam, SGD, RMSprop


# In[70]:


from sklearn.metrics import confusion_matrix


# In[71]:


import tensorflow as tf
import os
import numpy as np
import pandas as pd
import glob
import matplotlib as plt
import matplotlib.image as mpimg


# In[72]:


#visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[73]:


DATASET_DIR = "C:/Users/Khalai V/Desktop/COVID/dataset"


# In[74]:


os.listdir(DATASET_DIR)


# In[75]:


#creates list of normal -- need pillow to process images outside of .png
normal_images = []

for img_path in glob.glob(DATASET_DIR + '/normal/*'):
    normal_images.append(mpimg.imread(img_path))
fig = plt.figure()
fig.suptitle('normal')
plt.imshow(normal_images[0], cmap = 'gray')

covid_images = []

for img_path in glob.glob(DATASET_DIR + '/covid/*'):
    covid_images.append(mpimg.imread(img_path))
    
fig = plt.figure()
fig.suptitle('covid')
plt.imshow(covid_images[0], cmap = 'gray')
    


# In[76]:


print(len(normal_images))
print(len(covid_images))


# In[77]:


IMG_W = 150
IMG_H = 150
CHANNELS = 3

INPUT_SHAPE = (IMG_W, IMG_H, CHANNELS)
NB_CLASSES = 2
EPOCHS = 48
BATCH_SIZE = 6


# In[78]:



model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(250, (3,3)))
model.add(Activation('relu'))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(AvgPool2D(2,2))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(AvgPool2D(2,2))

model.add(Conv2D(256,(2,2)))
model.add(Activation('relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[79]:


model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


# In[80]:


model.summary()


# In[81]:


#from sklearn.preprocessing import LabelEncoder
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split = 0.3)
train_generator = train_datagen.flow_from_directory(DATASET_DIR, target_size=(IMG_H, IMG_W), batch_size=BATCH_SIZE, class_mode = 'binary', subset = 'training')
validation_generator = train_datagen.flow_from_directory(DATASET_DIR, target_size = (IMG_H, IMG_W), batch_size=BATCH_SIZE, class_mode = 'binary', shuffle = False, subset = 'validation')
history = model.fit_generator(train_generator, steps_per_epoch = train_generator.samples // BATCH_SIZE, validation_data = validation_generator, validation_steps = validation_generator.samples // BATCH_SIZE, epochs=EPOCHS)


# In[82]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[83]:


print('training_accuracy', history.history['accuracy'][-1])
print('validation_accuracy', history.history['val_accuracy'][-1])


# In[84]:


label = validation_generator.classes


# In[85]:


pred = model.predict(validation_generator)
predicted_class_indices = np.argmax(pred,axis = 1)
labels = (validation_generator.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
predictions = [labels2[k] for k in predicted_class_indices]
print(predicted_class_indices)
print (labels)
print (predictions)


# In[86]:


cf = confusion_matrix(predicted_class_indices, label)
cf


# In[89]:


exp_series = pd.Series(label)
pred_series = pd.Series(predicted_class_indices)
pd.crosstab(exp_series, pred_series, rownames=['Actual'], colnames = ['Predicted'], margins = True)


# In[92]:


plt.matshow(cf)
plt.title('Confusion Matrix Plot')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[ ]: