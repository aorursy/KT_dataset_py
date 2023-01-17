#Importing Necessary Libraries.

from PIL import Image

import numpy as np

import os

import cv2

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
print(os.listdir('../input/cell_images/cell_images'))
infectados = os.listdir('../input/cell_images/cell_images/Parasitized/')

saudaveis = os.listdir('../input/cell_images/cell_images/Uninfected/')
data = []

labels = []



#we'll save 3 images from each image, one rotated 30º and another rotated 60° from the original



for i in infectados:

    try:

    

        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((50 , 50))

        rotated30 = resize_img.rotate(30)

        rotated60 = resize_img.rotate(60)

        blur = cv2.blur(np.array(resize_img) ,(10,10))

        data.append(np.array(resize_img))

        data.append(np.array(rotated30))

        data.append(np.array(rotated60))

        data.append(np.array(blur))

        labels.append(1)

        labels.append(1)

        labels.append(1)

        labels.append(1)

        

    except AttributeError:

        print('')

    

for s in saudaveis:

    try:

        

        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+s)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((50 , 50))

        rotated30 = resize_img.rotate(30)

        rotated60 = resize_img.rotate(60)

        data.append(np.array(resize_img))

        data.append(np.array(rotated30))

        data.append(np.array(rotated60))

        labels.append(0)

        labels.append(0)

        labels.append(0)

        

    except AttributeError:

        print('')
celulas = np.array(data)

classes = np.array(labels)

import matplotlib.pyplot as plt

plt.figure(1 , figsize = (15 , 9))

n = 0 

for i in range(49):

    n += 1 

    r = np.random.randint(0 , celulas.shape[0] , 1)

    plt.subplot(7 , 7 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(celulas[r[0]])

    plt.title('{} : {}'.format('Infectados' if classes[r[0]] == 1 else 'Saudaveis' ,

                               classes[r[0]]) )

    plt.xticks([]) , plt.yticks([])

    

plt.show()
plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

plt.imshow(celulas[0])

plt.title('Infectada')

plt.xticks([]) , plt.yticks([])



plt.subplot(1 , 2 , 2)

plt.imshow(celulas[60000])

plt.title('Saudavel')

plt.xticks([]) , plt.yticks([])



plt.show()
np.random.seed(0)



n = np.arange(celulas.shape[0])

np.random.shuffle(n)

cells = celulas[n]

labels = classes[n]
cells = cells.astype(np.float32)

labels = labels.astype(np.int32)

cells = cells/255
from sklearn.model_selection import train_test_split



X , x_test , y , y_test = train_test_split(cells , labels , 

                                            test_size = 0.2 ,

                                            random_state = 111)



X_tr , X_val , y_tr , y_val = train_test_split(X , y , 

                                                    test_size = 0.5 , 

                                                    random_state = 111)
num_classes=len(np.unique(labels))

len_data=len(cells)



y_tr = keras.utils.to_categorical(y_tr,num_classes)

y_val = keras.utils.to_categorical(y_val,num_classes)

y_test = keras.utils.to_categorical(y_test,num_classes)
print(y_tr.shape, y_val.shape, X_tr.shape, X_val.shape)
#creating the architecture of our neural network 



model = Sequential()



model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))



model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))



model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))



model.add(Conv2D(filters=128,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))



model.add(Dropout(0.2))

model.add(Flatten())



model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))



model.add(Dense(2,activation="sigmoid")) # we're using 0 to healthy cells and 1 to infected cells 

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_tr,y_tr,validation_data=(X_val, y_val),batch_size=50,epochs=10,verbose=1)
score = model.evaluate(x_test, y_test, verbose=1)
pred = model.predict(x_test, verbose=1)
from keras.models import load_model

model.save('cells.h5')
# Plot training & validation accuracy values

plt.figure(figsize=(10,10))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.figure(figsize=(10,10))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.summary()
from keras.models import Model

layer_outputs = [layer.output for layer in model.layers]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(X_tr[20].reshape(1,50,50,3))

 

def display_activation(activations, col_size, row_size, act_index): 

    activation = activations[act_index]

    activation_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')

            activation_index += 1
plt.imshow(X_tr[20][:,:,:]);
display_activation(activations, 4, 4, 0)
display_activation(activations, 4, 4, 1)
display_activation(activations, 4, 4, 2)
display_activation(activations, 4, 4, 3)
display_activation(activations, 4, 4, 4)
from sklearn import metrics

conf = metrics.confusion_matrix(y_test[:,0], np.around(pred[:,0]))

conf_norm = conf/conf.sum(axis=1)

print(conf_norm)
from keras.applications.mobilenet import MobileNet

mobile_model = MobileNet(input_shape=(50,50,3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling=None, classes=2)
x = mobile_model.output

x = Flatten()(x)

preds = Dense(2,activation='softmax')(x) #final layer with softmax activation

mobile_model=Model(inputs=mobile_model.input,outputs=preds)

mobile_model.summary()
mobile_model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

history_mobile = mobile_model.fit(X_tr,y_tr,validation_data=(X_val, y_val),batch_size=128,epochs=10,verbose=1)
score_mobile = mobile_model.predict(x_test, verbose=1)
conf_mob = metrics.confusion_matrix(y_test[:,0], np.around(score_mobile[:,0]))

conf_mob_norm = conf_mob/conf_mob.sum(axis=1)

print(conf_mob_norm)
layer_outputs = [layer.output for layer in mobile_model.layers[1:]]

activation_model = Model(inputs=mobile_model.input, outputs=layer_outputs)

activations_mob = activation_model.predict(X_tr[20].reshape(1,50,50,3))

plt.imshow(X_tr[20][:,:,:]);

display_activation(activations_mob, 8, 4, 1)
display_activation(activations_mob, 8, 4, 2)
display_activation(activations_mob, 8, 4, 3)
display_activation(activations_mob, 8, 8, 10)
display_activation(activations_mob, 8, 8, 20)
# Plot training & validation accuracy values

plt.figure(figsize=(10,10))

plt.plot(history_mobile.history['acc'])

plt.plot(history_mobile.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.figure(figsize=(10,10))

plt.plot(history_mobile.history['loss'])

plt.plot(history_mobile.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.evaluate(x_test, y_test, verbose=1)
mobile_model.evaluate(x_test, y_test, verbose=1)