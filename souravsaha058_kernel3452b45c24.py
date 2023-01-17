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
import numpy as np
import pandas as pd 
from os import getcwd
import matplotlib.pyplot as plt
import os
import tensorflow as tf

filename = "/kaggle/input/facial-expression/fer2013/fer2013.csv"
names=['emotion','pixels','usage']
df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)
im=df['pixels']
def get_data(filename):
    x=[]
    y=[]
    tmp = True
    for line in open(filename):
        if tmp:
            tmp = False
        else:
            row = line.split(",")
            y.append(int(row[0]))
            x.append([int(p) for p in row[1].split( )])
    x,y = np.array(x)/255.0 , np.array(y)
    return x , y
            
x,y = get_data(filename)
num_class = len(set(y))
print(num_class)
n ,d =x.shape
x= x.reshape(n,48,48,1)
from sklearn.model_selection import train_test_split
x_train,x_test , y_train ,y_test = train_test_split(x,y,test_size =.2,random_state = 0 )

model = tf.keras.models.Sequential([
    # Note the input shape is 28 x 28 grayscale.
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(7, activation='softmax') # 26 alphabets/hand-signs so 26 classes!
])
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale =1/255)

print(x_train.shape)
print(x_test.shape)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'rmsprop',metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint("model_weights.h5",monitor = 'val_accuracy', save_weights_only = True , mode = 'max',verbose =1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,patience = 2 , min_lr = 0.00001, mode = 'auto')
callbacks =  [ checkpoint , reduce_lr]
history=model.fit(x=x_train,     
            y=y_train, 
            batch_size=64,
            epochs=20, 
            steps_per_epoch=len(x_train) / 64,
            verbose=1, 
            validation_data=(x_test,y_test),
            validation_steps=len(x_test) / 64,
            callbacks= callbacks
           )
model.evaluate(x_test,y_test)

h=model.fit(x=x_train,     
            y=y_train, 
           batch_size=64, 
            epochs=20, 
            verbose=1, 
            validation_data=(x_test,y_test)
           )
model.evaluate(x_test,y_test)
import matplotlib.pyplot as plt
#plot the chart for accuracy ans loss on both training and validation
%matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs,acc, 'r' , label = "Training accuracy")
plt.plot(epochs,val_acc , 'b' , label = "validation accuracy")
plt.title(' training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs , loss , 'r', label = "training loss")
plt.plot(epochs , val_loss , 'b', label = "validation loss")
plt.title(' training and validation loss')
plt.legend()
plt.figure()

plt.show()
