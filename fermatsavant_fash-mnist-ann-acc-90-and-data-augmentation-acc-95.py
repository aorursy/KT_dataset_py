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
#package
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
from skimage import io
#Loading data
df_train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
y_train = df_train.label.to_numpy()
X_train = np.asarray(df_train.iloc[:, 1:]).reshape([-1,28,28,1])
df_test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
y_test = df_test.label.to_numpy()
X_test = np.asarray(df_test.iloc[:, 1:]).reshape([-1,28,28,1])
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
rows = 5 # defining no. of rows in figure
cols = 6 # defining no. of colums in figure

f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 

for i in range(rows*cols): 
    f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration
    plt.imshow(X_train[i].reshape([28,28]),cmap="Blues") 
    plt.axis("off")
    plt.title(str(y_train[i]), y=-0.15,color="green")
plt.savefig("digits.png")
#Normalization
X_train = X_train/255
X_test = X_test/255
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),#Input layer
keras.layers.Dense(1568, activation="relu"),#Hidden layers
keras.layers.Dense(564, activation="relu"),#Hidden layers
keras.layers.Dense(350, activation="relu"),#Hidden layers
keras.layers.Dense(110, activation="relu"),#Hidden layers

keras.layers.Dense(10, activation="softmax")#Output layers
])
initial_lr = 0.1
loss = "sparse_categorical_crossentropy"
model.compile(SGD(lr=initial_lr), loss=loss ,metrics=['accuracy'])
model.summary()
#model.compile(loss="sparse_categorical_crossentropy",
#optimizer="sgd",
#metrics=["accuracy"], initial_lr = 0.001)

x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)
batch_size = 32
history = model.fit(x_train, y_train, epochs=30,batch_size=batch_size,
                    validation_data=(x_valid, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

val_p = np.argmax(model.predict(X_test),axis =1)

error = 0
confusion_matrix = np.zeros([10,10])
for i in range(X_test.shape[0]):
    confusion_matrix[y_test[i],val_p[i]] += 1
    if y_test[i]!=val_p[i]:
        error +=1
        
print("Confusion Matrix: \n\n" ,confusion_matrix)
print("\nErrors in validation set: " ,error)
print("\nError Persentage : " ,(error*100)/val_p.shape[0])
print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])
f = plt.figure(figsize=(10,8.5))
f.add_subplot(111)

plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")
plt.colorbar()
plt.tick_params(size=5,color="white")
plt.xticks(np.arange(0,10),np.arange(0,10))
plt.yticks(np.arange(0,10),np.arange(0,10))

threshold = confusion_matrix.max()/2 

for i in range(10):
    for j in range(10):
        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")
        
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("Confusion_matrix1.png")
plt.show()
x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback  


batch_size = 128
epochs = 5
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

lrr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5, min_lr=0.1)

history_da = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                 validation_data = (x_valid, y_valid), steps_per_epoch=len(x_train)/batch_size, 
                                 epochs=epochs,shuffle=True)#, callbacks=[lrr])
pd.DataFrame(history_da.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

