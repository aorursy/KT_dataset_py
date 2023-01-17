

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import tensorflow as tf

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, SpatialDropout2D, Conv2D, MaxPooling2D, AveragePooling1D, Reshape

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler



import matplotlib.pyplot as plt

print(os.listdir('../input'))
x_train = pd.read_csv('../input/train.csv')

x_train.head()
# set up training data and labels

dim_x = 28

dim_y = 28

batch_size=32



# read in data/labels

x_train.shape

y_train = np.array(x_train['label'])

x_train.drop('label', axis = 1, inplace = True)

x_train = np.array(x_train.values)



print("data shapes", x_train.shape, y_train.shape, "classes: ",len(np.unique(y_train)))



classes = len(np.unique(y_train))

x_train = x_train.reshape((-1, dim_x,dim_y,1))

# convert labels to one-hot

print(np.unique(y_train))

y = np.zeros((np.shape(y_train)[0],len(np.unique(y_train))))



# convert index labels to one-hot

for ii in range(len(y_train)):

    #print(y_train[ii])

    y[ii,y_train[ii]] = 1

y_train = y
# split into training/validation

no_validation = int(0.1 * (x_train.shape[0]))



x_val = x_train[0:no_validation,...]

y_val = y_train[0:no_validation,...]



x_train = x_train[no_validation:,...]

y_train = y_train[no_validation:,...]



print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)



# define image generators with mild augmentation

train_datagen = ImageDataGenerator(rescale = 1./255,\

                                   rotation_range=30,\

                                   width_shift_range=0.025,\

                                   height_shift_range=0.025,\

                                   shear_range=0.35,\

                                   zoom_range=0.075)



train_generator = train_datagen.flow(x=x_train,\

                                     y=y_train,\

                                     batch_size=batch_size,\

                                     shuffle=True)



test_datagen = ImageDataGenerator(rescale=1./255)



val_generator = test_datagen.flow(x=x_val,\

                                    y=y_val,\

                                    batch_size=batch_size,\

                                    shuffle=True)
# define model AlexNet (but topless)

model = Sequential()



model.add(Conv2D(filters=96, kernel_size=(5,5), strides=1,input_shape=(dim_x,dim_y,1), activation=tf.nn.relu))

model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(filters=256, kernel_size=(5,5), strides=1, activation=tf.nn.relu))

#model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1, activation=tf.nn.relu))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1, activation=tf.nn.relu))

model.add(SpatialDropout2D(rate=0.67))

model.add(Conv2D(filters=250, kernel_size=(3,3), strides=1, activation=tf.nn.relu))

model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())

model.add(Reshape((250,1)))

model.add(AveragePooling1D(pool_size=25,strides=25))

model.add(Reshape(([10])))

model.add(Activation(tf.nn.softmax))



model.summary()



model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
def learning_schedule(epoch):

    if epoch <= 1:

        lr = 3e-4

    elif epoch <= 10:

        lr = 1e-5

    elif epoch <= 50:

        lr = 3e-6

    elif epoch <= 150:

        lr = 1e-6

    else:

        lr = 1e-8

    return lr



# callbacks

lrate = LearningRateScheduler(learning_schedule)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=600, verbose=1, mode='auto')
steps_per_epoch = int(len(y_train)/batch_size)



max_epochs = 4096



history = model.fit_generator(generator=train_generator,\

                                steps_per_epoch=steps_per_epoch,\

                                validation_data=val_generator,\

                                validation_steps=50,\

                                epochs=max_epochs,\

                                callbacks=[early, lrate],\

                                verbose=2)
plt.figure(figsize=(15,12))

plt.subplot(211)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Accuracy and Loss",fontsize=28)

plt.ylabel('accuracy',fontsize=24)

plt.legend(['Train','Val'],fontsize=18)



plt.subplot(212)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('epoch',fontsize=24)

plt.ylabel('loss',fontsize=24)

plt.legend(['Train','Val'],fontsize=18)

plt.show()
x_test = pd.read_csv('../input/test.csv')

x_test.head()



x_test = np.array(x_test.values)

x_test = x_test / 255.



print("data shape", x_test.shape)



x_test = x_test.reshape((-1, dim_x,dim_y,1))

# predict!

y_pred = model.predict(x_test)
# visualize success (?) :/



def imshow_w_labels(img,  pred,count):

    plt.subplot(1,4,count+1)

    plt.imshow(img, cmap="gray")

    plt.title("Prediction: %i, "%(pred), fontsize=24)

    

    

count = 0

mask = [1,3,3,7]

plt.figure(figsize=(24,6))

for kk in range(50,600):

    

    if y_pred[kk,:].argmax() == mask[count]:

        imshow_w_labels(x_test[kk,:,:,0],y_pred[kk,...].argmax(), count)

        count += 1

    if count >= 4: break

plt.show()


# convert one-hot predictions to indices

results = np.argmax(y_pred,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)