import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import cv2                 
import numpy as np         
import os                  
from random import shuffle
from tqdm import tqdm      

train_dir = '../input/train'
test_dir = '../input/test'
def get_label(img):
    label = img.split('.')[0]
    if label == 'cat': 
        return [1,0]
    elif label == 'dog': 
        return [0,1]
def making_train_data():
    training_data = []
    
    for img in tqdm(os.listdir(train_dir)):
        label = get_label(img)
        path = os.path.join(train_dir,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50,50))
        training_data.append([np.array(img),np.array(label)])
        
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
def making_test_data():
    testing_data = []
    
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir , img)
        img_num = img.split('.')[0]
        img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img , (50,50))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
train_data = making_train_data()
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
train = train_data[0:20000]
test = train_data[20000:25000]
print(len(train) , len(test))
X = np.array([i[0] for i in train]).reshape(-1,1,50,50)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,1,50,50)
test_y = [i[1] for i in test]
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.0,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  

datagen.fit(X)
from keras.callbacks import ReduceLROnPlateau
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
Y = np.asarray(Y)
Y.reshape(len(Y) , 2)
test_y = np.asarray(test_y)
test_y.reshape(len(test_y) , 2)
test_x = test_x.reshape(-1, 1, 50, 50)
test_x = test_x / 255
X = X / 255
X.shape
from keras.layers import Dense , Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D , BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
def swish_activation(x):
    '''
    Keras implementation of swish https://arxiv.org/abs/1710.05941
    '''
    return (K.sigmoid(x) * x)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(1,50,50)))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation=swish_activation))
model.add(Dropout(0.4))
model.add(Dense(2 , activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
batch_size = 128
epochs = 20


model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
steps_per_epoch = len(train_data) // batch_size
validation_steps = len((test_x, test_y)) // batch_size
history = model.fit_generator(datagen.flow(X, Y, batch_size=batch_size),
                    steps_per_epoch=X.shape[0] // batch_size,
                    callbacks=[lr_reduce],
                    validation_data=(test_x, test_y),
                    epochs = epochs, verbose = 2)
score = model.evaluate(test_x, test_y, verbose=0)
print('valid loss:', score[0])
print('valid accuracy:', score[1])
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_data = making_test_data()
temp = np.array(test_data)
with open('final.csv','w') as f:
    f.write('id,label\n')
            
with open('final.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(1,1,50,50)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))
