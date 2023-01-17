# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
dir_list=[]

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        dir_list.append(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(dir_list) # 24335 Total pictures
pred_list=[]  # collecting all pred data in same list

for i in dir_list:
    if 'seg_pred' in i:
        pred_list.append(i)
        

        
test_list=[]  # collecting all pred data in same list

for i in dir_list:
    if 'seg_test' in i:
        test_list.append(i)
        
        
        
        
        
train_list=[]  # ccollecting all data in same list

for i in dir_list:
    if 'seg_train' in i:
        train_list.append(i)
len(train_list)  # 14034 train images
len(test_list) #3000 test images
len(pred_list)  # 7301 pred images
train_label=[]

a=0
b=1
c=2
d=3
e=4
f=5

for i in train_list:
    if 'buildings' in i:
        train_label.append(a)
    elif 'forest' in i:
        train_label.append(b)
    elif 'glacier' in i:
        train_label.append(c)
    elif 'mountain' in i:
        train_label.append(d)
    elif 'sea' in i:
        train_label.append(e)
    elif 'street' in i :
        train_label.append(f)
        
        
test_label=[]

a=0
b=1
c=2
d=3
e=4
f=5

for i in test_list:
    if 'buildings' in i:
        test_label.append(a)
    elif 'forest' in i:
        test_label.append(b)
    elif 'glacier' in i:
        test_label.append(c)
    elif 'mountain' in i:
        test_label.append(d)
    elif 'sea' in i:
        test_label.append(e)
    elif 'street' in i :
        test_label.append(f)
#buildings,street,mountain,glacier,sea,forest
df=pd.DataFrame({"Label_train":train_label})
df2=pd.DataFrame({"Label_test":test_label})
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,5))
sns.countplot("Label_train",data=df)
plt.xlabel("Class Distrubition") # I think class distrubition is balanced
plt.figure(figsize=(5,5))
sns.countplot("Label_test",data=df2)
plt.xlabel("Class Distrubition") 


plt.figure(figsize=(16,16))

for i in range(25):
    
    img = cv2.imread(train_list[i])
    plt.subplot(5,5,(i%25)+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Label_train'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16)) 

for i in range(3000,3025):
    
    img = cv2.imread(train_list[i])
    plt.subplot(5,5,(i%25)+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Label_train'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16)) 

for i in range(5000,5025):
    
    img = cv2.imread(train_list[i])
    plt.subplot(5,5,(i%25)+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Label_train'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16)) 

for i in range(9000,9025):
    
    img = cv2.imread(train_list[i])
    plt.subplot(5,5,(i%25)+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Label_train'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16)) 

for i in range(10025,10050):
    
    img = cv2.imread(train_list[i])
    plt.subplot(5,5,(i%25)+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Label_train'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16)) 

for i in range(12000,12025):
    
    img = cv2.imread(train_list[i])
    plt.subplot(5,5,(i%25)+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Label_train'].iloc[i])
    )
plt.show()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dir_train='/kaggle/input/intel-image-classification/seg_train/seg_train/'

dir_test='/kaggle/input/intel-image-classification/seg_test/seg_test/'


train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        dir_train, 
        target_size=(150,150), 
        batch_size=32,
        class_mode='categorical')



validation_generator = validation_datagen.flow_from_directory(
        dir_test,  
        target_size=(150, 150), 
        batch_size=8,
        class_mode='categorical')





import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization

model=tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150,3)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Dropout(0.25),
    
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
BatchNormalization(),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.25),

tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
BatchNormalization(),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.25),
    
#tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#tf.keras.layers.MaxPooling2D(2,2),

    
tf.keras.layers.Flatten(),
tf.keras.layers.Dropout(0.50),
tf.keras.layers.Dense(128, activation='relu'),
BatchNormalization(),
tf.keras.layers.Dropout(0.50),
tf.keras.layers.Dense(256),
tf.keras.layers.Dense(6, activation='softmax')])

model.summary()
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get(['acc']>0.99)):
            print("Reached desired acc")
            self.model.stop_training=True
            
callback=myCallback()
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr=0.001),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

Model = model.fit_generator(
      train_generator,
      steps_per_epoch=439,  
      epochs=30,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8
)
History=Model
%matplotlib inline
acc = History.history['accuracy']
val_acc = History.history['val_accuracy']
loss = History.history['loss']
val_loss = History.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
