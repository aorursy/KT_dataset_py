import os

import sys

import numpy as np

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from keras import Sequential

from keras.preprocessing import image

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from IPython.display import SVG

from keras.layers import Dense, Conv2D, MaxPooling2D,Dropout,BatchNormalization,Flatten
parapath = '../input/cell_images/cell_images/Parasitized/'

uninpath = '../input/cell_images/cell_images/Uninfected/'

parastized = os.listdir(parapath)

uninfected = os.listdir(uninpath)

print(sys.getsizeof(parastized))

print(sys.getsizeof(parastized))
data = []

label = []

for para in parastized:

    try:

        img =  image.load_img(parapath+para,target_size=(112,112))

        x = image.img_to_array(img)

        data.append(x)

        label.append(1)

    except:

        print("Can't add "+para+" in the dataset")

for unin in uninfected:

    try:

        img =  image.load_img(uninpath+unin,target_size=(112,112))

        x = image.img_to_array(img)

        data.append(x)

        label.append(0)

    except:

        print("Can't add "+unin+" in the dataset")
data = np.array(data)/255

label = np.array(label)
print(sys.getsizeof(data))

print(data.shape)
x_train, x_test, y_train, y_test = train_test_split(data,label,test_size = 0.1,random_state=0)
def MalariaModel():

    model = Sequential()

    model.add(Conv2D(filters = 4, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a11', input_shape = (112, 112, 3)))  

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a12'))

    model.add(BatchNormalization(name = 'a13'))

    #input = (112,112,4)

    model.add(Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a21'))   

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a22'))

    model.add(BatchNormalization(name = 'a23'))

    #input = (56,56,8)

    model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a31'))   

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a32'))

    model.add(BatchNormalization(name = 'a33'))

    #input = (28,28,16)

    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same', name = 'a41'))   

    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), name = 'a42'))

    model.add(BatchNormalization(name = 'a43'))

    #input = (14,14,32)

    model.add(Flatten())

    model.add(Dense(32, activation = 'relu', name = 'fc1'))

    model.add(Dense(1, activation = 'sigmoid', name = 'prediction'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
modelone = MalariaModel()

modelone.summary()
output = modelone.fit(x_train, y_train,validation_split=0.1,epochs=4, batch_size=50)
preds = modelone.evaluate(x = x_test,y = y_test)

print("Test Accuracy : %.2f%%" % (preds[1]*100))
plt.plot(output.history['acc'])

plt.plot(output.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'cross-validation'], loc='upper left')

plt.savefig('Accuracy.png',dpi=100)

plt.show()
modelone.save('malariaCNNModel.h5')
modelpic = plot_model(modelone, to_file='model.png')

SVG(model_to_dot(modelone).create(prog='dot', format='svg'))