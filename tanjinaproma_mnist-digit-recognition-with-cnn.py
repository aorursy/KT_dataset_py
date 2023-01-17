# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train  = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test  = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(" Train Shape: ",train.shape,"\n Test Shape: ", test.shape)

# Creating X_train & y_train
X = train.drop(['label'], axis = 1)
print("\n X Shape: ", X.shape)
y = train['label']


X = X.values # Converting to array from dataframe
# X = X/255.0 # Normalize data
X = X.reshape(-1, 28, 28, 1)  # Reshape X_train into 3 dimensions 


print("\n X Shape: ", X.shape)
print("\n y Shape: ", y.shape)
y = to_categorical(y, num_classes = 10)
#y = np.array(y).astype(float)
print(y.shape, y[0])
plt.imshow(X[88][:,:,0])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

train_datagen = ImageDataGenerator(
            rescale=1/255.0,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
           
    )

validation_datagen = ImageDataGenerator(
            rescale=1/ 255.0
)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=100
)

validation_generator = validation_datagen.flow(
    X_test, 
    y_test,
    batch_size=100
)


#Callback Class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.999):
            print("\nReached  99.9% accuracy & stopping training!!!")
            self.model.stop_training = True
                
callbacks = myCallback()    
def plot(history):
    %matplotlib inline
    import matplotlib.pyplot as plt
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
    

    
# Model 1:  
    
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3),padding = 'Same', activation = 'relu', input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3),padding = 'Same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3),padding = 'Same', activation = 'relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
    
model1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# history = model1.fit(X_train, y_train, epochs = 50, validation_data = (X_test, y_test), callbacks= [callbacks])
   
#model.summary()

history = model1.fit(
                train_generator, 
                epochs = 50,
                validation_data = validation_generator,
                callbacks = [callbacks]
               
)
plot(history)
#  Model 2
model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5),padding = 'Same', activation = 'relu', input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (5,5),padding = 'Same', activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (5,5),padding = 'Same', activation = 'relu'),
    tf.keras.layers.Conv2D(256, (3,3),padding = 'Same', activation = 'relu'),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
    
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# history = model.fit(X_train, y_train, epochs = 50, callbacks= [callbacks])
history_2 = model_2.fit_generator(
                train_generator, 
                epochs = 50,
                validation_data = validation_generator,
                callbacks = [callbacks]
               
)
plot(history_2)
test_dataframe = test
test = test.values
test= test/255.0
test= test.reshape(-1, 28, 28, 1)

y_pred = model6.predict(test)
y_pred = np.argmax(y_pred, axis =1)

submission = pd.DataFrame({
    "ImageID": test_dataframe.index+1,
    "Label": y_pred
})


submission.to_csv('my_submission.csv', index=False)