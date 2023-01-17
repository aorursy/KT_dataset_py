# Importing the required libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

os.listdir('/kaggle/input/')
#Reading the train data - It has image information along with categories

filenames_train = pd.read_csv('/kaggle/input/avships/data/train.csv')
filenames_train.head()
print(filenames_train['category'].value_counts())
sns.countplot(x='category' , data=filenames_train)
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(filenames_train['image'],filenames_train['category'],train_size = 0.8,random_state = 102,stratify = filenames_train['category'])
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
#Reading test data
filenames_test = pd.read_csv('/kaggle/input/avships/data/test.csv')
filenames_test.head()
temp_test = list(filenames_test['image'])
temp_test[:6]
X_train = list(X_train)
X_val = list(X_val)
X_train[99]
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array,load_img,array_to_img
#Reading sample image
img = load_img('/kaggle/input/avships/data/images/'+X_train[99])
img
#Function to read ,resize images
num_channels = 3
img_width = 128
img_height = 128
images_train = np.ndarray(shape = (len(X_train),img_height,img_width,num_channels),dtype = np.float16)
images_val = np.ndarray(shape = (len(X_val),img_height,img_width,num_channels),dtype = np.float16)
images_test = np.ndarray(shape = (len(temp_test),img_height,img_width,num_channels),dtype = np.float16)

def read_images(fname,array_name):
  for i in range(len(fname)):
    img = load_img('/kaggle/input/avships/data/images/'+fname[i],color_mode = 'rgb',target_size = (img_height, img_width))
    img = img_to_array(img)
    img = img/255
    array_name[i] = img
    if i % 1000 == 0:
      print("%d images read" %i)
  print("All images read")
#Reading train images
read_images(X_train,images_train)
#Reading val images
read_images(X_val,images_val)
#Reading test images
read_images(temp_test,images_test)
#label encoding
num_classes = 5
y_train_1 = to_categorical(y_train-1,num_classes = num_classes,dtype='float16')
y_val_1 = to_categorical(y_val-1,num_classes = num_classes,dtype = 'float16')
#Using pre trained models - Inception
from keras.applications import InceptionV3
base_model = InceptionV3(include_top = False)

base_model.summary()
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,BatchNormalization
from keras.layers import Activation, Dense
y = base_model.output
y = GlobalAveragePooling2D()(y)
y= Dense(300,activation='relu')(y)
y= BatchNormalization()(y)
pred_inception= Dense(5,activation='softmax')(y)
model3= Model(inputs=base_model.input,outputs=pred_inception)
#making convolution layers untrainable
for layer in base_model.layers:
    layer.trainable=True
#checking trainable and non-trainable  parameters
model3.summary()
#compiling the model
model3.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
hist2 = model3.fit(x = images_train,y = y_train_1,batch_size = 32, epochs = 15,validation_data = (images_val,y_val_1))
# Plot training & validation accuracy values
plt.plot(hist2.history['accuracy'])
plt.plot(hist2.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(hist2.history['loss'])
plt.plot(hist2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
pred = model3.predict(images_test,batch_size=32)
classes_list = []
for i in range(len(pred)):
    classes_list.append(np.argmax(pred[i])+1)
classes_list[:4]
#We have used the test dataframe and appended the predicted classes
filenames_test['category'] = pd.DataFrame(data=classes_list)
filenames_test.head()
#Saving the predictions for submission
filenames_test.to_csv(path_or_buf='submission_4.csv',header=True,index=False)
#model3.save_weights('inceptionV3.hdf5')

#Reading sample image
#{'Cargo': 1, 'Military': 2, 'Carrier': 3, 'Cruise': 4, 'Tankers': 5}

i=4

ftemp=filenames_test[(filenames_test['image']==str(temp_test[i]))]
print(ftemp.to_string(index=False))
img = load_img('/kaggle/input/avships/data/images/'+temp_test[i])
img