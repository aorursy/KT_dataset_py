# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt #for plotting things
import cv2
from PIL import Image
import tensorflow as tf 

# Keras Libraries
import random
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras import optimizers

tf.random.set_seed(2)
np.random.seed(1)

base_dir = ('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images')
Uninfected_dir=os.path.join(base_dir,'Uninfected')
par_dir=os.path.join(base_dir,'Parasitized')
print(len(os.listdir(Uninfected_dir)))
print(len(os.listdir(par_dir)))
for i in range (1,8):
    rand_norm= np.random.randint(0,len(os.listdir(Uninfected_dir)))
    unin_pic = os.listdir(Uninfected_dir)[rand_norm]
    
    print('uninfected title: ',unin_pic)


    #print(rand_norm)


    norm_address=os.path.join(Uninfected_dir,unin_pic)

    #print(norm_address)

    norm_load = Image.open(norm_address) #Loading image

    #f = plt.figure(figsize= (10,10))
    img_plot = plt.imshow(norm_load,cmap='gray') #showing image
    print(norm_load.size)
    i=i+1

#Data input

IMG_SIZE=64
CATEGORIES = ['Parasitized', 'Uninfected']
dataset = []

def generate_data():
    for category in CATEGORIES:
        path = f'../input/cell-images-for-detecting-malaria/cell_images/{category}'
        class_id = CATEGORIES.index(category)
        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image))
                image_array = cv2.resize(image_array, (IMG_SIZE , IMG_SIZE),3)
                dataset.append([image_array, class_id])
            except Exception as e:
                print(e)
                
    random.shuffle(dataset)
                


datas = []
labels = []
for img in os.listdir(par_dir):
    try:
        img_read = plt.imread('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/' + img)
        img_resize = cv2.resize(img_read, (50, 50),3)
        img_array = np.array(img_resize)
        datas.append([img_array,1])
        
    except Exception as e:
                print(e)
        
for img in os.listdir(Uninfected_dir):
    try:
        img_read =plt.imread('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/' + img)
        img_resize = cv2.resize(img_read, (50, 50),3)
        img_array = np.array(img_resize)
        datas.append([img_array,0])
        
        labels.append(0)
    except Exception as e:
                print(e)
random.shuffle(datas)

data = []
labels = []
for features, label in datas:
    data.append(features)
    labels.append(label)
print(labels)    
    

#Data preprocess

data = np.array(data)
labels=np.array(labels)
#data.reshape(data.shape[0], 50, 50, 3)

print(data.shape)
train_data, og_data, train_labels, og_labels = train_test_split(data, 
                                                          labels,
                                                          test_size=0.2,random_state=101
                                                                   )

validation_data,test_data , validation_labels,test_labels = train_test_split(og_data, 
                                                                    og_labels,
                                                                    test_size=0.2,random_state=101)
print(train_labels.shape)
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape)
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(50,50,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    
    
     tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    
    tf.keras.layers.Flatten(),
   
    tf.keras.layers.Dense(128, activation='relu'), 
     tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')  
    
])

model.summary()
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen= ImageDataGenerator(rescale=1./255)
train_genarator=train_datagen.flow(
train_data,
train_labels,
batch_size=48
)

valid_genarator=validation_datagen.flow(
validation_data,
validation_labels,
batch_size=30

)
opt=tf.keras.optimizers.RMSprop(
    learning_rate=0.0001 
)

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
              
              metrics=['accuracy'])

history=model.fit(train_genarator,validation_data=valid_genarator,epochs=10,shuffle=False)
x=model.predict(test_data)



test_datagen=ImageDataGenerator(rescale=1./255)
test_gen=test_datagen.flow(test_data,test_labels,batch_size=1,shuffle=False)
print(x[0:10])
print(test_labels[0:10])
model.evaluate(test_gen)
x=model.predict_classes(test_data)
print(x[0:10])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training Acc.')
plt.plot(epochs, val_acc, 'b', label='Validation Acc.')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

