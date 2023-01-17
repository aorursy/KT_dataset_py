from __future__ import print_function, division
from builtins import range, input
# keras libraries
import keras
from keras.layers import *
from keras.models import Model,Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
# sklearn and matplotlibg for visualisation
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from glob import glob
import cv2
train_path='/kaggle/input/fruit-images-for-object-detection/data/train'
val_path='/kaggle/input/fruit-images-for-object-detection/data/test'
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(val_path + '/*/*.jp*g')
print("Number of Images for Training: ",len(image_files))
print("Number of Images for validating: ",len(glob(val_path + '/*/*.jp*g')))

# useful for getting number of classes
folders = glob(train_path + '/*')
print("Number of classes: ",len(folders))

# look at a random image 
plt.imshow(image.load_img(np.random.choice(image_files)))

plt.show()
# re-size all the images to 100x100
IMAGE_SIZE = [224, 224] 

# using the VGG16 model but not including the final output layer by using the command (include_top=False).
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# additional layers
x = Flatten()(vgg.output)
# we can add additional fully connected layers like this.
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()
# compliling the model.
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)

# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)
# training config:
epochs = 10
batch_size = 32

# create generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=False,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  val_path,
  target_size=IMAGE_SIZE,
  shuffle=False,
  batch_size=batch_size,
)
# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)
# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
import os 
import pandas as pd
import cv2
from PIL import Image
data=[]
labels=[]
filenames = os.listdir("/kaggle/input/fruit-images-for-object-detection/data/train/banana")
for filename in filenames:
    labels.append(0)
    img=cv2.imread('/kaggle/input/fruit-images-for-object-detection/data/train/banana/'+filename)
    img = Image.fromarray(img, 'RGB')
    size_image = img.resize((224, 224))
    data.append(np.array(size_image))
    files.append(filename)

filenames = os.listdir("/kaggle/input/fruit-images-for-object-detection/data/train/mixed")
for filename in filenames:
    labels.append(1)
    img=cv2.imread('/kaggle/input/fruit-images-for-object-detection/data/train/mixed/'+filename)
    img = Image.fromarray(img, 'RGB')
    size_image = img.resize((224, 224))
    data.append(np.array(size_image))
    files.append(filename)
    
filenames = os.listdir("/kaggle/input/fruit-images-for-object-detection/data/train/apple")
for filename in filenames:
    labels.append(2)  
    img=cv2.imread('/kaggle/input/fruit-images-for-object-detection/data/train/apple/'+filename)
    img = Image.fromarray(img, 'RGB')
    size_image = img.resize((224, 224))
    data.append(np.array(size_image))
    files.append(filename)
    
filenames = os.listdir("/kaggle/input/fruit-images-for-object-detection/data/train/orange")
for filename in filenames:
    labels.append(3)
    img=cv2.imread('/kaggle/input/fruit-images-for-object-detection/data/train/orange/'+filename)
    img = Image.fromarray(img, 'RGB')
    size_image = img.resize((224, 224))
    data.append(np.array(size_image))
    files.append(filename)

data=np.array(data)
labels=np.array(labels)
data1=data
s=np.arange(data.shape[0])
np.random.shuffle(s)
data=data[s]
labels=labels[s]
num_classes=len(np.unique(labels))
len_data=len(data)
(x_train,x_test)=data[(int)(0.1*len_data):],data[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)
(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=IMAGE_SIZE + [3]))
model1.add(MaxPooling2D(2,2))
model1.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model1.add(MaxPooling2D(2,2))
model1.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model1.add(MaxPooling2D(2,2))
model1.add(Flatten())
model1.add(Dense(100, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(4, activation='softmax'))

model1.summary()
model1.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model1.fit(x_train,y_train,batch_size=20,epochs=20,verbose=1)
# accuracies
plt.plot(model1.history.history['accuracy'], label='train acc')
plt.plot(model1.history.history['loss'], label='train loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
out=model.predict(train_generator)
out1=model1.predict(train_generator)
out
out1
rows=240
columns=4
y_predicted1=[]
for i in range(rows):
    max=-1
    for j in range(columns):
        if(max<out1[i][j]):
            max=out1[i][j]
            max1=j
    y_predicted1.append(max1)
data1=data1.astype('float32')/255
y_predicted1=model1.predict_classes(data1)
y_predicted1
rows=240
columns=4
y_predicted=[]
for i in range(rows):
    max=-1
    for j in range(columns):
        if(max<out[i][j]):
            max=out[i][j]
            max1=j
    y_predicted.append(max1)
y_predicted1=np.asarray(y_predicted)
y_predicted1
import os
import pandas as pd
files=[]
categories = []
filenames = os.listdir("/kaggle/input/fruit-images-for-object-detection/data/train/banana")
for filename in filenames:
    categories.append(0)
    files.append(filename)

filenames = os.listdir("/kaggle/input/fruit-images-for-object-detection/data/train/mixed")
for filename in filenames:
    categories.append(1)
    files.append(filename)
    
filenames = os.listdir("/kaggle/input/fruit-images-for-object-detection/data/train/apple")
for filename in filenames:
    categories.append(2)    
    files.append(filename)
    
filenames = os.listdir("/kaggle/input/fruit-images-for-object-detection/data/train/orange")
for filename in filenames:
    categories.append(3)
    files.append(filename)


df = pd.DataFrame({
    'filename': files,
    'category': categories
})
# from the VGG16 model
report = classification_report(df['category'], y_predicted)
print(report)
# from the simple model.
report = classification_report(df['category'], y_predicted1)
print(report)

