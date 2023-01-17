# Import all the required libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')



from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, BatchNormalization, Reshape 

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



reduce_lr = ReduceLROnPlateau(monitor= 'val_loss',

                             factor= 0.2,

                             patience= 2,

                             min_lr = 0.00001)
# load data

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
# prepare data

X_train = train.drop(['label'], axis= 1).values.reshape(-1, 28, 28, 1)

y_train = to_categorical(train['label'].values)



X_test = test.values.reshape(-1, 28, 28, 1)
print('shape of train set: ',X_train.shape)

print('shape of test set: ',X_test.shape)
model = Sequential()

model.add(Conv2D(32, kernel_size= 3, padding= 'same',\

                 activation= 'relu', input_shape= (28, 28, 1)))

model.add(MaxPool2D(2))

model.add(Flatten())

model.add(Dense(256, activation= 'relu'))

#output

model.add(Dense(10, activation= 'softmax'))



# compile the model

model.compile(loss= 'categorical_crossentropy', metrics= ['accuracy'],\

             optimizer= 'adam')



# fit the model

epochs = 10

history = model.fit(X_train, y_train, callbacks= [reduce_lr],\

          validation_split= 0.2, epochs= epochs, batch_size= 80)



# plot learning curves

plt.plot(np.arange(1, epochs+ 1), history.history['val_accuracy'])

plt.plot(np.arange(1, epochs+ 1), history.history['accuracy'])

plt.ylim(0.97, 1)
# Build CNN

num = 3

model = [0]* 3



for i in range(num):

    model[i] = Sequential()

    model[i].add(Conv2D(32, kernel_size= 3, activation= 'relu',\

                       input_shape= (28, 28, 1), padding= 'same'))

    model[i].add(MaxPool2D(2))

    if i > 0:

        model[i].add(Conv2D(32, kernel_size= 3, activation= 'relu', padding= 'same'))

        model[i].add(MaxPool2D(2))

    if i > 1:

        model[i].add(Conv2D(32, kernel_size= 3, activation= 'relu', padding= 'same'))

        model[i].add(MaxPool2D(2))

    

    model[i].add(Flatten())

    model[i].add(Dense(256, activation= 'relu'))

    model[i].add(Dense(10, activation= 'softmax'))

    

    # compile model

    model[i].compile(loss= 'categorical_crossentropy', 

                    optimizer= 'adam',

                    metrics= ['accuracy'])
# fit the models



epochs= 15

history = [0]*num



for j in range(num):

    history[j] = model[j].fit(X_train, y_train, validation_split= 0.3,\

            epochs= epochs, callbacks= [reduce_lr],verbose=2, batch_size= 80)

    print('Model {} trained'.format(j+1))
# plot learning curves



for i in range(num):

    plt.plot(np.arange(1, epochs + 1), history[i].history['val_accuracy'], label= '{}_conv_Layer'.format(i+1))

plt.xlabel('Epochs')

plt.ylabel('Validation Accuracy')

plt.title('Model comparison on validation set accuracy')

plt.legend(loc= 'upper left')

plt.ylim(0.96, 1)

plt.show()
# Build CNN

num = 4

model = [0]*num



for i in range(num):

    model[i] = Sequential()

    

    model[i].add(Conv2D(2**(i+3), kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1),\

                       padding= 'same'))

    model[i].add(MaxPool2D(2))

    

    model[i].add(Conv2D(2**(i+4), kernel_size= 3, activation= 'relu', padding= 'same'))

    model[i].add(MaxPool2D(2))

    

    model[i].add(Flatten())

    model[i].add(Dense(256, activation= 'relu'))

    model[i].add(Dense(10, activation= 'softmax'))

    

    # compile model

    model[i].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
# fit the models

history = [0]*num

epochs= 15



for j in range(num):

    history[j] = model[j].fit(X_train, y_train, validation_split= 0.3,\

            epochs= epochs, callbacks= [reduce_lr],verbose=0, batch_size= 80)

    print('Model {} trained'.format(j+1))
# plot learning curves

for i in range(num):

    plt.plot(np.arange(1, epochs + 1), history[i].history['val_accuracy'],\

             label= '{}_{}_CNN'.format(2**(i+3), 2**(i+4)))

plt.xlabel('Epochs')

plt.ylabel('Validation Accuracy')

plt.title('Model comparison on validation set accuracy')

plt.legend(loc= 'upper left')

plt.ylim(0.96, 1)

plt.show()
# Build CNN

num= 4

model = [0]*num



for i in range(num):

    model[i] = Sequential()

    

    model[i].add(Conv2D(32, kernel_size= 3 + i*2, activation= 'relu', input_shape= (28, 28, 1),\

                       padding= 'same'))

    model[i].add(MaxPool2D(2))

    

    model[i].add(Conv2D(64, kernel_size= 3 + i*2, activation= 'relu', padding= 'same'))

    model[i].add(MaxPool2D(2))

    

    model[i].add(Flatten())

    model[i].add(Dense(256, activation= 'relu'))

    model[i].add(Dense(10, activation= 'softmax'))

                 

    # Compile

    model[i].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

    
# fit the data

history = [0]*num

epochs= 15



for j in range(num):

    history[j] = model[j].fit(X_train, y_train, validation_split= 0.3,\

            epochs= epochs, callbacks= [reduce_lr],verbose=0, batch_size= 80)

    print('Model {} trained'.format(j+1))
# plot learing curves

for i in range(num):

    plt.plot(np.arange(1, epochs + 1), history[i].history['val_accuracy'],\

             label= '{}x{}_kernel'.format(3 + i*2, 3 + i*2))

plt.xlabel('Epochs')

plt.ylabel('Validation Accuracy')

plt.title('Model comparison on validation set accuracy')

plt.legend(loc= 'upper left')

plt.ylim(0.96, 1)

plt.show()
# Build CNN

num = 3

model = [0]*num



for i in range(num):

    model[i] = Sequential()

    

    model[i].add(Conv2D(32, kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1), padding= 'same'))

    model[i].add(MaxPool2D(2))

    

    model[i].add(Conv2D(64, kernel_size= 3, activation= 'relu', padding= 'same'))

    model[i].add(MaxPool2D(2))

    

    model[i].add(Flatten())

    model[i].add(Dense(128, activation= 'relu'))

    

    if i > 0:

        model[i].add(Dense(128, activation= 'relu'))

    if i > 1:

        model[i].add(Dense(128, activation= 'relu'))

        

    model[i].add(Dense(10, activation= 'softmax'))

    

    # compile

    model[i].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

    
# fit the data

history = [0]*num

epochs= 15



for j in range(num):

    history[j] = model[j].fit(X_train, y_train, validation_split= 0.3, callbacks= [reduce_lr],\

                             epochs= epochs, batch_size= 80, verbose= 0)

    print('Model {} is trained'.format(j+1))
# plot learing curves

for i in range(num):

    plt.plot(np.arange(1, epochs + 1), history[i].history['val_accuracy'],\

             label= '{}_Dense_layer'.format(i+1))

plt.xlabel('Epochs')

plt.ylabel('Validation Accuracy')

plt.title('Model comparison on validation set accuracy')

plt.legend(loc= 'upper left')

plt.ylim(0.96, 1)

plt.show()
# Build CNN

num = 4

model = [0]*4



for i in range(num):

    model[i] = Sequential()

    

    model[i].add(Conv2D(32, kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1), padding= 'same'))

    model[i].add(MaxPool2D(2))

    

    model[i].add(Conv2D(64, kernel_size= 3, activation= 'relu', padding= 'same'))

    model[i].add(MaxPool2D(2))

    

    model[i].add(Flatten())

    model[i].add(Dense(2**(i+5), activation= 'relu'))

    model[i].add(Dense(10, activation= 'softmax'))

    

    # compile

    model[i].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

    
# fit the data

history = [0]*num

epochs= 15



for j in range(num):

    history[j] = model[j].fit(X_train, y_train, validation_split= 0.3, callbacks= [reduce_lr],\

                             epochs= epochs, batch_size= 80, verbose= 0)

    print('Model {} is trained'.format(j+1))
# plot learing curves

for i in range(num):

    plt.plot(np.arange(1, epochs + 1), history[i].history['val_accuracy'],\

             label= '{}_Dense_layer'.format(2**(i+5)))

plt.xlabel('Epochs')

plt.ylabel('Validation Accuracy')

plt.title('Model comparison on validation set accuracy')

plt.legend(loc= 'upper left')

plt.ylim(0.96, 1)

plt.show()
# Build CNN

num = 7

model = [0]*num



for i in range(num):

    model[i] = Sequential()

    

    model[i].add(Conv2D(32, kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1), padding= 'same'))

    model[i].add(MaxPool2D(2))

    model[i].add(Dropout(i*0.1))

    

    model[i].add(Conv2D(64, kernel_size= 3, activation= 'relu', padding= 'same'))

    model[i].add(MaxPool2D(2))

    model[i].add(Dropout(i*0.1))

    

    model[i].add(Flatten())

    model[i].add(Dense(256, activation= 'relu'))

    model[i].add(Dropout(i*0.1))

    model[i].add(Dense(10, activation= 'softmax'))

    

    # compile

    model[i].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
# fit the data

history = [0]*num

epochs= 20



for j in range(num):

    history[j] = model[j].fit(X_train, y_train, validation_split= 0.3, callbacks= [reduce_lr],\

                             epochs= epochs, batch_size= 80, verbose= 0)

    print('Model {} is trained'.format(j+1))
# plot learning curve

for i in range(num):

    plt.plot(np.arange(1, epochs + 1), history[i].history['val_accuracy'],\

             label= '{}%_Dropout'.format(i*10))

plt.xlabel('Epochs')

plt.ylabel('Validation Accuracy')

plt.title('Model comparison on validation set accuracy')

plt.legend(loc= 'upper left')

plt.ylim(0.96, 1)

plt.show()
# Build CNN

num = 5

model= [0]*num

history = [0]*num

epochs= 20



# Basic model



model[0] = Sequential()

    

model[0].add(Conv2D(32, kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1), padding= 'same'))

model[0].add(MaxPool2D(2))

model[0].add(Dropout(0.1))



model[0].add(Conv2D(64, kernel_size= 3, activation= 'relu', padding= 'same'))

model[0].add(MaxPool2D(2))

model[0].add(Dropout(0.1))



model[0].add(Flatten())

model[0].add(Dense(256, activation= 'relu'))

model[0].add(Dropout(0.1))

model[0].add(Dense(10, activation= 'softmax'))



# compile

model[0].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])





# fit data

history[0] = model[0].fit(X_train, y_train, validation_split= 0.3, callbacks= [reduce_lr],\

                             epochs= epochs, batch_size= 80, verbose= 2)
# Model with Data augmentation

X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size= 0.3)



datagen = ImageDataGenerator(featurewise_center= False,

                             samplewise_center= False,

                             featurewise_std_normalization= False,

                             samplewise_std_normalization= False,

                             zca_whitening= False,

                             rotation_range= 10,

                             zoom_range = 0.1,

                             width_shift_range= 0.1,

                             height_shift_range= 0.1,

                             horizontal_flip= False,

                             vertical_flip= False)



datagen.fit(X_train)
model[1] = Sequential()

    

model[1].add(Conv2D(32, kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1), padding= 'same'))

model[1].add(MaxPool2D(2))

model[1].add(Dropout(0.1))



model[1].add(Conv2D(64, kernel_size= 3, activation= 'relu', padding= 'same'))

model[1].add(MaxPool2D(2))

model[1].add(Dropout(0.1))



model[1].add(Flatten())

model[1].add(Dense(256, activation= 'relu'))

model[1].add(Dropout(0.1))

model[1].add(Dense(10, activation= 'softmax'))



# compile

model[1].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])



# fit data

history[1] = model[1].fit(datagen.flow(X_train2, y_train2, batch_size= 80),validation_data= (X_val2, y_val2),\

                         steps_per_epoch= X_train2.shape[0]//80, epochs= epochs, callbacks= [reduce_lr],\

                         verbose= 2)
# model with batch normalization 

model[2] = Sequential()

    

model[2].add(Conv2D(32, kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1), padding= 'same'))

model[2].add(MaxPool2D(2))

model[2].add(BatchNormalization())

model[2].add(Dropout(0.1))



model[2].add(Conv2D(64, kernel_size= 3, activation= 'relu', padding= 'same'))

model[2].add(MaxPool2D(2))

model[2].add(BatchNormalization())

model[2].add(Dropout(0.1))



model[2].add(Flatten())

model[2].add(Dense(256, activation= 'relu'))

model[2].add(BatchNormalization())

model[2].add(Dropout(0.1))

model[2].add(Dense(10, activation= 'softmax'))



# compile

model[2].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
# fit the data

history[2] = model[2].fit(X_train, y_train, validation_split= 0.3,\

                         batch_size= 80, epochs= epochs, callbacks= [reduce_lr],\

                         verbose= 2)
# model with no pooling layers

model[3] = Sequential()

    

model[3].add(Conv2D(32, kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1), padding= 'same'))

model[3].add(Conv2D(32, kernel_size= 3, activation= 'relu', strides= 2, padding= 'same'))

model[3].add(Dropout(0.1))



model[3].add(Conv2D(64, kernel_size= 3, activation= 'relu', padding= 'same'))

model[3].add(Conv2D(64, kernel_size= 3, activation= 'relu', strides= 2, padding= 'same'))

model[3].add(Dropout(0.1))



model[3].add(Flatten())

model[3].add(Dense(256, activation= 'relu'))

model[3].add(Dropout(0.1))

model[3].add(Dense(10, activation= 'softmax'))



# compile

model[3].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
# fit data

history[3] = model[3].fit(X_train, y_train, validation_split= 0.3,\

                         batch_size= 80, epochs= epochs, callbacks= [reduce_lr],\

                         verbose= 2)
# model with all advance concepts

model[4] = Sequential()

    

model[4].add(Conv2D(32, kernel_size= 3, activation= 'relu', input_shape= (28, 28, 1), padding= 'same'))

model[4].add(BatchNormalization())

model[4].add(Conv2D(32, kernel_size= 3, activation= 'relu', strides= 2, padding= 'same'))

model[4].add(BatchNormalization())

model[4].add(Dropout(0.1))



model[4].add(Conv2D(64, kernel_size= 3, activation= 'relu', padding= 'same'))

model[4].add(BatchNormalization())

model[4].add(Conv2D(64, kernel_size= 3, activation= 'relu', strides= 2, padding= 'same'))

model[4].add(BatchNormalization())

model[4].add(Dropout(0.1))



model[4].add(Flatten())

model[4].add(Dense(256, activation= 'relu'))

model[4].add(BatchNormalization())

model[4].add(Dropout(0.1))

model[4].add(Dense(10, activation= 'softmax'))



# compile

model[4].compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
# fit augmented data

history[4] = model[4].fit(datagen.flow(X_train2, y_train2, batch_size= 80),validation_data= (X_val2, y_val2),\

                         steps_per_epoch= X_train2.shape[0]//80, epochs= epochs, callbacks= [reduce_lr],\

                         verbose= 2)
# plot learning curves

labels = ['Basic model',

         'DA',

         'BN',

         'Pool~Conv',

         'Combined']

for i in range(num):

    plt.plot(np.arange(1, epochs + 1), history[i].history['val_accuracy'],\

             label= labels[i])

plt.xlabel('Epochs')

plt.ylabel('Validation Accuracy')

plt.title('Model comparison on validation set accuracy')

plt.legend(loc= 'upper left')

plt.ylim(0.96, 1)

plt.show()
# predict with combined model

epochs= 25



model[4].fit_generator(datagen.flow(X_train,y_train, batch_size=64), epochs = epochs, 

    steps_per_epoch = X_train.shape[0]//64, callbacks=[reduce_lr], verbose=0)



# SUBMIT

results = model[4].predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST_SUB.csv",index=False)