# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

img_dir=[]

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        img_dir.append(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(img_dir)
labels=[]

for i in img_dir:
    if 'tulip' in i:
        labels.append("tulip")
    elif 'daisy' in i :
        labels.append("daisy")
    elif 'rose' in i:
        labels.append("rose")
    elif 'dandelion' in i:
        labels.append("dandelion")
    elif 'sunflower' in i:
        labels.append("sunflower")
        
        
df=pd.DataFrame({"Labels":labels})
df.head()
plt.figure(figsize=(10,5))
sns.countplot("Labels",data=df) # Tulip and dandelion are more than other categories
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape = (128, 128, 3), 
                                include_top = False, 
                                weights = 'imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False
    
    
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.optimizers import Adam

x = layers.Flatten()(last_output)
x = layers.Dense(256, activation='relu')(x)
x=layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x=layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense  (5, activation='softmax')(x)

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.001),
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
plt.figure(figsize=(16,16))

for i in range(25):
    img = cv2.imread(img_dir[i])
    plt.subplot(5,5,(i%25)+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Labels'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16))

for i in range(1000,1025):
    img = cv2.imread(img_dir[i])
    plt.subplot(5,5,(i%25)+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Labels'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16))

for i in range(2100,2125):
    img = cv2.imread(img_dir[i])
    plt.subplot(5,5,(i%25)+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Labels'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16))

for i in range(3000,3025):
    img = cv2.imread(img_dir[i])
    plt.subplot(5,5,(i%25)+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Labels'].iloc[i])
    )
plt.show()
plt.figure(figsize=(16,16))

for i in range(4000,4025):
    img = cv2.imread(img_dir[i])
    plt.subplot(5,5,(i%25)+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.xlabel(
        "Class:"+str(df['Labels'].iloc[i])
    )
plt.show()
img_list=[]
label_list=[]




for i in os.listdir("/kaggle/input/flowers-recognition/flowers/flowers/dandelion"):
    
    if i!="flickr.py" and i!="run_me.py" and i!="flickr.pyc": # those cause a problem so I need to filter them 
    
        path="/kaggle/input/flowers-recognition/flowers/flowers/dandelion/"+i
    
        img=cv2.imread(path)
        img = cv2.resize(img,(128,128))
        img_list.append(img)
    
        label=0 # 0 for dandelion
        label_list.append(label)
        
        
for i in os.listdir("/kaggle/input/flowers-recognition/flowers/flowers/tulip"):
    
    path="/kaggle/input/flowers-recognition/flowers/flowers/tulip/"+i
    
    img=cv2.imread(path)
    img = cv2.resize(img,(128,128))
    img_list.append(img)
        
    label=1 # 1 for tulip
    label_list.append(label)
    
    
for i in os.listdir("/kaggle/input/flowers-recognition/flowers/flowers/sunflower"):
    
    path="/kaggle/input/flowers-recognition/flowers/flowers/sunflower/"+i
    
    img=cv2.imread(path)
    img = cv2.resize(img,(128,128))
    img_list.append(img)
        
    label=2 # 2 for sunflower
    label_list.append(label)
        
for i in os.listdir("/kaggle/input/flowers-recognition/flowers/flowers/rose"):
    
    path="/kaggle/input/flowers-recognition/flowers/flowers/rose/"+i
    
    img=cv2.imread(path)
    img = cv2.resize(img,(128,128))
    img_list.append(img)
        
    label=3 # 3 for rose 
    label_list.append(label)
        
for i in os.listdir("/kaggle/input/flowers-recognition/flowers/flowers/daisy"):
    
    path="/kaggle/input/flowers-recognition/flowers/flowers/daisy/"+i
    
    img=cv2.imread(path)
    img = cv2.resize(img,(128,128))
    img_list.append(img)   
        
    label=4 #4 for daisy
    label_list.append(label)
        
    
    
        


len(img_list) # good 
len(label_list) # good 
img_list= np.array(img_list) # turn img_list into numpy array
from keras.utils.np_utils import to_categorical
labels = to_categorical(label_list,num_classes = 5)

labels[0] # Labels
img_list.shape # my image data is ready
X_train, X_test, y_train, y_test = train_test_split(img_list,labels, test_size=0.1, random_state=42)
print('X_train shape:', X_train.shape)
print('X_test shape: ',X_test.shape)
print('y_train shape: ',y_train.shape)
print('y_test shape: ',y_test.shape)
datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2, 
        height_shift_range=0.2,  
        horizontal_flip=True,
        vertical_flip=False)

datagen.fit(X_train)
batch_size = 64
epochs = 10
history = model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),epochs=epochs,validation_data=(X_test,y_test))
%matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

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
print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")
y_pred=model.predict(X_test)
y_pred[:5]
# İt is going to give array that has 5 list in itself.
#Each list has 5 numbers,This numbers are probabilty of classes.
#Example, if first number bigger than others in the list,This image could be class 0 so dandelion.
import seaborn as sns

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report

print(classification_report(Y_true, Y_pred_classes)) 

#0: dandelion 
#1: Tulip
#2: Sunflower
#3: Rose
#4: Daisy
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization

model=tf.keras.models.Sequential([
    
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3),padding='same'),
tf.keras.layers.MaxPooling2D(2,2),
BatchNormalization(),
tf.keras.layers.Dropout(0.25),
    
tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
tf.keras.layers.MaxPooling2D(2,2),
BatchNormalization(),
tf.keras.layers.Dropout(0.25),
    
tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding="same"),
tf.keras.layers.MaxPooling2D(2,2),
BatchNormalization(),
tf.keras.layers.Dropout(0.25),
    
tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding="same"),
tf.keras.layers.MaxPooling2D(2,2),
BatchNormalization(),
tf.keras.layers.Dropout(0.25),

    
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
BatchNormalization(),
tf.keras.layers.Dropout(0.5),
    
tf.keras.layers.Dense(512, activation='relu'),
BatchNormalization(),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(5, activation='softmax')])

model.summary()


from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr=0.001),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])



batch_size = 64
epochs = 30
history = model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),epochs = epochs, validation_data = (X_test,y_test))
%matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

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
print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")
y_pred=model.predict(X_test)
import seaborn as sns

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()