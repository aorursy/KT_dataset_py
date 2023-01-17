import os

# Ignore  the warnings

import matplotlib.pyplot as plt

import numpy as np

import cv2

import seaborn as sns

import random as rn

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm
a = '../input/flowers/flowers'
print (os.listdir(a))
classes = {0:'daisy',

           1:'dandelion',

           2:'rose',

           3:'sunflower',

           4:'tulip'}

img_height = 200

img_width = 200

input_shape = (200,200,3)

epochs = 10
train_folder = '../input/flowers/flowers/'
print (os.listdir(train_folder))
sns.barplot(x=['tulip','roses','dandelion','sunflower','daisy'] , y=[len(os.listdir(train_folder+'tulip')),

                                                                    len(os.listdir(train_folder+'rose')),

                                                                    len(os.listdir(train_folder+'dandelion')),

                                                                    len(os.listdir(train_folder+'sunflower')),

                                                                    len(os.listdir(train_folder+'daisy'))])
training_data=[]

label = []

def process_image():

    for i in tqdm(range(len(classes))):

        print ('Working on directory {}'.format(classes[i]))

        for j in os.listdir(train_folder+'//'+classes[i]):

                if j.endswith("jpg"):

                    img_read = cv2.imread(os.path.join(train_folder+'/'+classes[i]+'/'+j), cv2.IMREAD_COLOR)

                    img_read = cv2.resize(img_read,(img_height,img_width))

                    training_data.append(np.array(img_read))

                    label.append(str(classes[i]))

                else:

                    continue

               # print (os.path.join(train_folder,classes[i]+'/'+j))
process_image()
plt.figure(figsize=(10,10))

for i in range(10):

    plt.subplot(5,5,i+1)

    l=rn.randint(0,len(label)-1)

    plt.imshow(cv2.cvtColor(training_data[l], cv2.COLOR_BGR2RGB))

    plt.title(label[l])

    plt.tight_layout()

plt.show()

#plt.tight_layout()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

Y = encoder.fit_transform(label)

Y = to_categorical(Y,len(classes))
X = np.array(training_data)
X = X/255.0
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam , RMSprop

# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization , GlobalAveragePooling2D

from keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint





model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',input_shape=(input_shape),activation ='relu'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 512, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dense(len(classes), activation = "softmax"))
model.summary()
learning_rate=0.00001
#Call backs

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    patience=2,

    cooldown=1,

    min_lr=0.000001,

    verbose=1)



early_stopping = EarlyStopping(monitor='val_loss', patience=2 , verbose=1)





#Optimizer

optimizer=Adam(lr=learning_rate)



callbacks =[reduce_lr,early_stopping]
model.compile(optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=3,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
History = model.fit_generator(datagen.flow(x_train,y_train),

                              epochs = epochs, validation_data = (x_test,y_test),

                               steps_per_epoch=x_train.shape[0],callbacks=callbacks ,verbose=2)
#Model history

accuracy = model.history.history['acc']

val_accuracy= model.history.history['val_acc']

loss = model.history.history['loss']

val_loss = model.history.history['val_loss']



#Plot accuracy

plt.title('Accuracy / Val Accuracy')

plt.plot(accuracy , label='Accuracy')

plt.plot(val_accuracy,label='Val accuracy')

plt.legend()
#Plot loss

plt.title('Loss / Validation loss')

plt.plot(loss,label='Loss')

plt.plot(val_loss, label='Validation loss')

plt.legend()
import tensorflow as tf
def predict(img):

    label = model.predict(img.reshape(-1,200,200,3))

    return classes[np.argmax(label)]
plt.figure(figsize=(20,20))

for i in range(10):

    l=rn.randint(0,len(x_test)-1)

    predict_label = predict(x_test[l])

    plt.subplot(5,5,i+1)

    plt.title('Pre :{} Tru :{}'.format(predict_label,classes[np.argmax(y_test[l])]))

    plt.imshow(x_test[l])

plt.tight_layout()