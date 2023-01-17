import numpy as np

from numpy import genfromtxt



from keras import layers



from keras.layers import (Input, Dense, Activation, ZeroPadding2D,

BatchNormalization, Flatten, Conv2D, concatenate)



from keras.layers import (AveragePooling2D, MaxPooling2D, Dropout,

GlobalMaxPooling2D, GlobalAveragePooling2D)



from keras.models import Model, load_model

from keras import regularizers, optimizers



from keras.utils import to_categorical,plot_model



import matplotlib.pyplot as plt
def get_data(directory):

    

    train = genfromtxt(directory,delimiter=',')

    train = np.delete(train,(0),axis=0)

    train_y = train[:,0]

    train_y = np.reshape(train_y,[train_y.shape[0],1])

    train_x = np.delete(train,(0),axis=1)

    

    return train_x,train_y
train_x,train_y = get_data('../input/fashion-mnist_train.csv')

test_x,test_y = get_data('../input/fashion-mnist_test.csv')



train_x = np.reshape(train_x,[train_x.shape[0],28,28,1])

train_x = train_x/255



test_x = np.reshape(test_x,[test_x.shape[0],28,28,1])

test_x = test_x/255



train_y = to_categorical(train_y)

test_y = to_categorical(test_y)



print("Shape of Train_x - ", train_x.shape)

print("Shape of Train_y - ", train_y.shape)



print("Shape of Test_x - ", test_x.shape)

print("Shape of Test_y - ", test_y.shape)

def model(input_size):

    

    weight_decay = 1e-4

    

    x_input = Input(shape=(input_size,input_size,1))

    x = ZeroPadding2D((2,2))(x_input)

    

    x = Conv2D(16,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization(axis=3)(x)

    x = Dropout(0.2)(x)

    

    x = Conv2D(32,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization(axis=3)(x)

    x = Dropout(0.2)(x)

    

    x = Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((2,2))(x)

    x = BatchNormalization(axis=3)(x)

    x = Dropout(0.2)(x)



    x = Flatten()(x)

    

    x = Dense(256,activation='relu')(x)

    x = Dropout(0.2)(x)

    

    x = Dense(10,activation='softmax')(x)

    

    model = Model(inputs=x_input,outputs=x,name='model')

    

    return model



model = model(28)



model.compile(loss='categorical_crossentropy',

             optimizer='Adadelta',

             metrics=['accuracy'])



hist = model.fit(train_x,train_y,batch_size=128,epochs=40,verbose=2,validation_data=(test_x,test_y),shuffle=True)



model.save("FashionMNIST")
pred = model.evaluate(test_x,test_y)

print("Accuracy on Test Set is ", pred[1])
plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()