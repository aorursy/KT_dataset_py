 %matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential,Model, load_model

from keras import losses

from keras import initializers

from keras.layers import Activation ,Dropout ,Flatten,Dense

from keras.layers.convolutional import Conv2D, MaxPooling2D ,ZeroPadding2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam,rmsprop

from keras import regularizers 

from keras.constraints import maxnorm

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.image as mpimg

import os
#Print Data Available

print(os.listdir("../input"))
# Load Test Data

csv_test = pd.read_csv('../input/sample_submission.csv')



# read test CSV 

csv_test.head(10)
# Load Train Data & Generate Labels

csv_train = pd.read_csv('../input/train_labels.csv')



# read training CSV

labels = pd.Series(csv_train['Category'])



csv_train.head(10)
# load train, test Data

x_train = np.load('../input/train_images.npy')

x_test = np.load('../input/test_images.npy')



# Print size of data

print(x_train.shape)

print(x_test.shape)
# scalling 

lb= LabelBinarizer()



y_train = labels

y_train=lb.fit_transform(y_train) 



x_train = x_train /255

x_test = x_test/255

classes = [ 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]



print('labels',y_train.shape) #after one hot so the shape is 50000*10 "10"each classes 0000100000 

print("classes: ", classes)
# Create Test & Train Data

X_train, X_valid, Y_train, Y_valid = train_test_split(

    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train

)
# Verify input Data

print("Training data set size: ", X_train.shape, Y_train.shape, "\n")

for a in range (10):

    print(X_train[a].shape, Y_train[a])
#VGG

np.random.seed(0)

weight_decay = 0.0005

def creat_model():

    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same',data_format="channels_first", input_shape=x_train.shape[1:]))

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3,3), padding='same',data_format="channels_first",))

    model.add(Activation('relu'))

    

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Conv2D(64, (3,3), padding='same',data_format="channels_first", kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), padding='same',data_format="channels_first", kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.3))



    model.add(Conv2D(128, (3,3), padding='same',data_format="channels_first", kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), padding='same',data_format="channels_first", kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.7))

    

    model.add(Conv2D(256, (3, 3), padding='same',data_format="channels_first",kernel_initializer='TruncatedNormal', kernel_regularizer=regularizers.l2(weight_decay))) 

    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3),data_format="channels_first",kernel_initializer='TruncatedNormal', kernel_regularizer=regularizers.l2(weight_decay)))

    model.add(Activation('relu'))

    model.add(BatchNormalization())

    

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.7)) 



    model.add(Flatten())

    

    model.add(Dense(num_classes, activation='softmax'))

    opt = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999) # try momentum 

    #opt=rmsprop(lr=0.001,decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics =['accuracy'])

    model.summary()

    return model 
batch_size = 128

num_classes = 10 

epochs =250



model =creat_model() 



cnn=model.fit(X_train, Y_train, batch_size=batch_size,epochs=epochs, validation_data=(X_valid,Y_valid),shuffle=True)

# Plots for training and testing process: loss and accuracy

 

plt.figure(0)

plt.plot(cnn.history['acc'],'b')

plt.plot(cnn.history['val_acc'],'g')

plt.xticks(np.arange(0, 101, 2.0))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Accuracy")

plt.title("Training Accuracy vs Validation Accuracy")

plt.legend(['train','validation'])

 

 

plt.figure(1)

plt.plot(cnn.history['loss'],'b')

plt.plot(cnn.history['val_loss'],'g')

plt.xticks(np.arange(0, 101, 2.0))

plt.rcParams['figure.figsize'] = (8, 6)

plt.xlabel("Num of Epochs")

plt.ylabel("Loss")

plt.title("Training Loss vs Validation Loss")

plt.legend(['train','validation'])

 

 



plt.show()
# Predict Against Testing Data

prediction =model.predict(x_test[:100],verbose=1) 

for p in prediction:

    print("Predicted Class:", p, "\n")
# Predict Against Training Data

index = 0

prediction =model.predict(x_train[:20],verbose=1) 

for p in prediction:

    print("Actual class", y_train[index])

    print("Predicted class:", p ,"\n")

    index += 1