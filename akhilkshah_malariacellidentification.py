import numpy as np

import cv2

import os

import matplotlib.pyplot as plt



import numpy as np

from numpy import genfromtxt



from keras import layers



from keras.layers import (Input, Dense, Activation, ZeroPadding2D,

BatchNormalization, Flatten, Conv2D, concatenate)



from keras.layers import (AveragePooling2D, MaxPooling2D, Dropout,

GlobalMaxPooling2D, GlobalAveragePooling2D)



from keras.models import Model, load_model

from keras import regularizers, optimizers



from sklearn.model_selection import train_test_split



from keras.utils import to_categorical
def create_train_data(directory):

    

    train_x = []

    train_y = []

    

    for each_class in os.listdir(directory):

        path = os.path.join(directory,each_class)



        for each_img in os.listdir(path):

            

            img_path = os.path.join(path,each_img)

            

            if each_img == 'Thumbs.db':

                continue

            

            img = cv2.imread(img_path)

            img = cv2.resize(img,(50,50))

            high_pass_filter = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])

            img_enhanced = cv2.filter2D(img, -1, high_pass_filter)

            

            train_x.append(img_enhanced)

            

            if each_class == 'Parasitized':

                label = 1

                train_y.append(label)

                

            elif each_class == 'Uninfected':

                label = 0

                train_y.append(label)

                

        

                

    return train_x,train_y

    
train_x,train_y = create_train_data('../input/cell_images/cell_images/')



train_x = np.array(train_x)

train_y = np.array(train_y)

train_y = np.reshape(train_y,[train_y.shape[0],1])



x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.30, shuffle= True)



print('Shape of x_train - ', x_train.shape)

print('Shape of y_train - ', y_train.shape)

print('Shape of x_test -  ', x_test.shape)

print('Shape of y_test -  ', y_test.shape)



x_train = x_train/255

x_test = x_test/255
def model(input_size):

    

    weight_decay = 1e-9

    

    x_input = Input(shape=(input_size,input_size,3))

    x = ZeroPadding2D((2,2))(x_input)

    

    x = Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization(axis=3)(x)

    x = Dropout(0.5)(x)

    

    x = Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization(axis=3)(x)

    x = Dropout(0.5)(x)

    

    x = Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization(axis=3)(x)

    x = Dropout(0.5)(x)

    

    x = Flatten()(x)

            

    x = Dense(1,activation='sigmoid')(x)

    

    model = Model(inputs=x_input,outputs=x,name='model')

    

    return model



model = model(50)



opt = optimizers.Adam()

model.compile(loss='binary_crossentropy',

             optimizer=opt,

             metrics=['accuracy'])



hist = model.fit(x_train,y_train,batch_size=256,epochs=20,verbose=2,validation_data=(x_test,y_test))

pred = model.evaluate(x_test,y_test)

print("Accuracy - ", pred[1])
plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()