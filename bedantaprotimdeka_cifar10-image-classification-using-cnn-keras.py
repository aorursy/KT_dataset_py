# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#importing libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.optimizers  import adam
from keras.callbacks import Callback

from keras.utils import np_utils


from keras.models import Sequential 
from keras.layers import Dense,Conv2D, MaxPooling2D, Dropout, Flatten
#loading the data
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# to know the type of data
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))
#take the first image look forom training data
x_train[0]
x_train= x_train/255
x_test = x_test/255
#normalize the pixel in the image -> 0 and 1
nClasses = 10
y_train = np_utils.to_categorical(y_train,nClasses)
y_test = np_utils.to_categorical(y_test,nClasses)
#to know thw shape 
print('x_train:',x_train.shape)
print('y_train:',y_train.shape)
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)
#creating the model
def createModel():
    model =Sequential()
    model.add(Conv2D(32, (3 ,3),padding = 'same',activation = 'relu',input_shape = (32,32,3)))
    model.add(Conv2D(32, (3, 3) ,activation = 'relu' ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    
    model.add(Conv2D(64, (3, 3), padding = 'same',activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu' ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    
    model.add(Conv2D(64, (3, 3), padding = 'same',activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu' ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    
    
    return model
model = createModel()
AdamOpt = adam(lr=0.001)                                                    #compile the model 
model.compile(optimizer=AdamOpt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()      
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs={} ):
        if(epoch % 5 == 0):
            print("just finished",epoch)
            print('--------------------------')                        #just a funtion to show to accuracy after 5 epochs
            print('losss = ',logs.get('val_loss'))
            print('accuracy :',logs.get('accuracy'))
            print('val_acc = ',logs.get('val_accuracy'))
            return    
batch_size = 256
epochs = 50                                           #fit the model with thw datasets
cc=CustomCallback()
hist = model.fit(x_train,y_train,batch_size =batch_size ,epochs =epochs,verbose=1,validation_data=(x_test, y_test),callbacks=[cc])
plt.figure(figsize=[8,6])
plt.plot(hist.history['accuracy'],linewidth=2.0)
plt.plot(hist.history['val_accuracy'],ls = '--')
plt.legend(['training accuracy','validation_accuracy'],fontsize=10,loc='lower right')          #Accuracy curve
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy curve')
plt.figure(figsize=[8,6])
plt.plot(hist.history['loss'],linewidth=2.0)
plt.plot(hist.history['val_loss'],ls = '--')
plt.legend(['training loss','validation_loss'],fontsize=10,loc='upper right')                                #loss curve
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Loss curve')
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize                        #load a new image for testing
imgg= imread('../input/thisistest/cat.jpg')
Image_resize = resize(imgg,(32,32))
img = plt.imshow(Image_resize)
#get the prob
import numpy as np
probabilities = model.predict(np.array([Image_resize]))
probabilities
# import numpy as np
number_of_class = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_of_class[index[9]])
model.save('cnn_cifar10.hs')  