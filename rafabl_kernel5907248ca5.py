import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D,Layer,Lambda,BatchNormalization
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data_p = np.load("../input/train_images_pure.npy") #arranging the pure train image into an array
train_data_r = np.load("../input/train_images_rotated.npy")
train_data_n = np.load("../input/train_images_noisy.npy")
train_data_b = np.load("../input/train_images_both.npy")
test_data = np.load("../input/Test_images.npy")
sample = pd.read_csv("../input/sample_sub.csv")


train_dt_p = train_data_p.reshape(train_data_p.shape[0], 28, 28 , 1).astype('float32')
train_dt_r = train_data_r.reshape(train_data_r.shape[0], 28, 28 , 1).astype('float32')
train_dt_n = train_data_n.reshape(train_data_n.shape[0], 28, 28 , 1).astype('float32')
train_dt_b = train_data_b.reshape(train_data_b.shape[0], 28, 28 , 1).astype('float32')
test_dt = test_data.reshape(test_data.shape[0],28,28,1).astype('float32')

train_labels = pd.read_csv("../input/train_labels.csv")
train_labels = np_utils.to_categorical(train_labels.loc[:,'label']).astype('int32')
train_labels.shape
train_labels
sample
def display_one(data,i, title): # displays the i-th image of the database
    plt.imshow(data[i]), plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()
display_one(train_data_p, 2,"Teste de Leitura de Imagem")
# Another useful function is one that shows two images at the same time, for comparison purposes
def display_two(db_i, db_j, i, j, title_i, title_j): # displays the i-th and the j-th image of two databases
    plt.subplot(121), plt.imshow(db_i[i]), plt.title(title_i)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(db_j[j]), plt.title(title_j)
    plt.xticks([]), plt.yticks([])
    plt.show()
display_two(train_data_p, train_data_n, 2, 2, "Third Pure Image","Third Noisy Image" )
display_two(train_data_p, train_data_r, 99, 99, "Hundreth Pure Image","Hundredth Rotated Image" )
display_two(train_data_p, train_data_r, 13, 13, "Thirteenth Pure Image","Thirteenth Rotated an Noisy Image" )

# ----------------------------------
# Remove noise
# Gaussian
def no_noise(data):
    n_noise = []
    for i in range(len(data)):
        blur = cv2.GaussianBlur(data[i], (5, 5), 0)
        n_noise.append(blur)
        return n_noise

train_data_p_blur = no_noise(train_data_p)[0]
train_data_n_blur = no_noise(train_data_n)[0]
train_data_r_blur = no_noise(train_data_r)[0]
train_data_r,train_data_r_blur
#display_two(train_data_p, train_data_p_blur, 2, 2, "Third Pure Image","Third Noisy Image" )
#---------------------------------
train_data_p[6].shape, train_data_p[6].dtype, train_data_p.shape, train_data_p[6].max()



from scipy import ndimage
sob = ndimage.sobel(train_data_p[2], mode='reflect')
plt.imshow(sob)

from keras import backend as K
model = Sequential()
K.set_image_data_format('channels_last')
model.add(Conv2D(64, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'  ))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='relu'  ))
model.add(Dropout(0.3))

model.add(Dense(10))
model.add(Activation("elu"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x = train_dt_r, y = train_labels, batch_size = 60,  epochs = 12)
scores = model.evaluate(train_dt_r, train_labels, verbose = 10 )
print ( scores )
predict= model.predict(test_dt).astype('int32')
predict = np.argmax(predict,axis = 1)
predict = pd.Series(predict, name="label")
predict.loc[:,'Id'] = sample.loc[:,'Id']
submission = pd.concat([pd.Series(range(0 ,9999) ,name = "Id"),   predict],axis = 1)
submission.to_csv("rotated.csv",index=False)
model.fit(x = train_dt_b, y = train_labels, batch_size = 60,  epochs = 12)
scores2 = model.evaluate(train_dt_b, train_labels, verbose = 10 )
print ( scores2 )
predict2 = model.predict(test_dt)
predict2 = np.argmax(predict2,axis = 1)
predict2 = pd.Series(predict2, name="label")
predict2.loc[:,'Id'] = sample.loc[:,'Id']
submission = pd.concat([pd.Series(range(0 ,9999) ,name = "Id"),   predict2],axis = 1)
submission.to_csv("Both.csv",index=False)

model2.add(Conv2D(64, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Conv2D(32, kernel_size=5, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model2.add(Flatten())
model2.add(Dense(units=100, activation='relu'  ))
model2.add(Dropout(0.2))
model2.add(Dense(units=100, activation='relu'  ))
model2.add(Dropout(0.3))

model2.add(Dense(10))
model2.add(Activation("elu"))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.fit(x = train_dt_r, y = train_labels, batch_size = 60,  epochs = 12)
score3 = model2.evaluate(train_dt_r, train_labels, verbose = 10 )
print ( scores3 )
scores3 = model2.evaluate(train_dt_b, train_labels, verbose = 10 )
print ( scores3 )
predict3 = model2.predict(test_dt)
predict3 = np.argmax(predict3,axis = 1)
predict3 = pd.Series(predict3, name="label")
predict3.loc[:,'Id'] = sample.loc[:,'Id']
submission = pd.concat([pd.Series(range(0 ,9999) ,name = "Id"),   predict3],axis = 1)
submission.to_csv("Rotated with Pooling.csv",index=False)
