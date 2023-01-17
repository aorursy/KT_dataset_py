import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D

mymodel=Sequential()
mymodel.add(Conv2D(12,kernel_size=(3,3), activation='relu', input_shape=(img_rows,img_cols,1)))
mymodel.add(Conv2D(12,kernel_size=(3,3), activation='relu'))
mymodel.add(Conv2D(12,kernel_size=(3,3), activation='relu') )  
mymodel.add(Flatten())
mymodel.add(Dense(100, activation='relu'))
mymodel.add(Dense(num_classes,activation='softmax'))
            
            
            

mymodel.compile(loss = keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy']
               )
mymodel.fit(x,y,
    batch_size=100,
           epochs=4,
           validation_split= 0.2)