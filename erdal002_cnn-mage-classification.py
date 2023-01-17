# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import cv2
import PIL
import tensorflow
from sklearn.metrics import confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
df2=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')


df_train=df1.copy()
df_test=df2.copy()
df_train.head()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.countplot("label",data=df_train)
plt.figure(figsize=(10,5))
sns.countplot("label",data=df_test)
y_train=df_train.label
y_test=df_test.label

X_train=df_train.drop('label',axis=1)
X_test=df_test.drop('label',axis=1)


plt.figure(figsize=(16,16))

for i in range(25):
    
    img = np.asarray(X_train.iloc[i])
    img = img.reshape((28,28))
    plt.subplot(5,5,(i%25)+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img,cmap='gray')
    plt.xlabel(
        "Class:"+str(df1['label'].iloc[i])
    )
plt.show()
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print('X_train shape:', X_train.shape)
print('X_test shape: ',X_test.shape)
print('y_train shape: ',y_train.shape)
print('y_test shape: ',y_test.shape)
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)


X=X/255.0
X_1=X_1/255.0

print('X_train shape:', X_train.shape)
print('X_test shape: ',X_test.shape)
print('y_train shape: ',y_train.shape)
print('y_test shape: ',y_test.shape)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(
      rotation_range=8,
      width_shift_range=0.08,
      height_shift_range=0.08,
      shear_range=False,
      zoom_range=0.1,
      horizontal_flip=True,
      vertical_flip=True,)


datagen.fit(X_train)
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization

model=tf.keras.models.Sequential([
    
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1),padding='same'),
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
tf.keras.layers.Dense(10, activation='softmax')])

model.summary()
batch_size = 128
epochs = 30
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr=0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
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
X_test=df_test.drop('label',axis=1)
y_test=df_test.label



X_test=X_test.values.reshape(-1,28,28,1)
X_test.shape
y_test=y_test.values.reshape(-1,1)
y_pred=model.predict_classes(X_test)
y_pred[:10] #  predictions
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
import seaborn as sns

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, linewidths=0.01,cmap="hot",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred)) # 

#0: t-shirt0: 
#1: Trouser;
#2: Pullover;
#3: Dress;
#4: Coat;
#5: Sandal;
#6: Shirt;
#7: Sneaker;
#8: Bag;
#9: Ankle boot.