import numpy as np 

import cv2                                         # working with, mainly resizing, images

import os,random                                          # dealing with directories

from keras.models import Sequential,Model                # creating sequential model of CNN

from keras.layers import Conv2D,BatchNormalization             # creating convolution layer

from keras.layers import MaxPooling2D              # creating maxpool layer

from keras.layers import Flatten,Activation                   # creating input vector for dense layer

from keras.layers import Dense,GlobalAveragePooling2D                     # create dense layer or fully connected layer

from keras.layers import Dropout                   # use to avoid overfitting by droping some parameters

import matplotlib.pyplot as plt   

from sklearn.model_selection import train_test_split   

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,Adadelta
IMG_WIDTH,IMG_HEIGTH=150,150

Batch_Size=8

Epoch=30



def onehotlabel(resim):

    if resim=="BOLT_LEFT":

        return 0

    elif resim=="BOLT_RIGHT":

        return 1 

    elif resim=="FASTENING_MODEL_LEFT":

        return 2

    elif resim=="FASTENING_MODEL_RIGHT":

        return 3 

    else: return -1

def generate_data(DATADIR):

    path = os.path.join(DATADIR)

    dataset = []

    for imge in os.listdir(DATADIR):        

        for img in os.listdir(os.path.join(DATADIR,imge)): 

            if img.endswith("jpg"):

                lbl=onehotlabel(imge)

                if (lbl!=-1):

                    im = cv2.imread(os.path.join(path,imge,img),cv2.IMREAD_GRAYSCALE)

                    im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGTH))

                    dataset.append([im,onehotlabel(imge)]) 

    

    random.shuffle(dataset)

    data = []

    labels = []

    for features, label in dataset:        

        data.append(features.astype('float32') / 255)

        labels.append(label)

    data = np.array(data)

#    data.reshape(data.shape[0], IMG_WIDTH ,IMG_HEIGTH,  1)

    train_data,test_data,train_labels,test_labels = train_test_split(data,labels,test_size=0.2)

    return train_data,test_data,train_labels,test_labels

path="../input/raildataset/RailDataSet"



train_data,test_data,train_labels,test_labels= generate_data(path)
train_data=train_data.reshape(-1,IMG_WIDTH ,IMG_HEIGTH,1)

train_data=train_data/255

test_data=test_data.reshape(-1,IMG_WIDTH ,IMG_HEIGTH,1)

test_data=test_data/255





datagen_train=ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



datagen_train.fit(train_data)
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same",input_shape=(IMG_WIDTH ,IMG_HEIGTH, 1)))

model.add(Activation("relu"))

model.add(BatchNormalization()) #Yığın normalleştirme derin sinir ağlarındaki herhangi bir katmana 0’a ortalanmış ve 1 ile 0 arasında değerlere sahip veriler vermemizi sağlar.

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))



model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))



model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation("softmax"))

model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

model.summary()
history = model.fit_generator(

    datagen_train.flow(train_data, train_labels, batch_size=Batch_Size),

    steps_per_epoch=len(train_data),

    validation_steps = len(test_data),

    epochs=Epoch,

    verbose = 1,

    validation_data=(test_data,test_labels)

    )
y_pred = model.predict_classes(test_data)

acc = np.sum(y_pred == test_labels) / np.size(y_pred)

print("Test accuracy = {}".format(acc))





final_loss, final_acc = model.evaluate(test_data, test_labels, verbose=1)

print("validation loss: {0:.6f}, validation accuracy: {1:.6f}".format(final_loss, final_acc))



accuracy = history.history['accuracy']

loss = history.history['loss']

val_accuracy = history.history['val_accuracy']

val_loss = history.history['val_loss']



print(f'Training Accuracy: {np.max(accuracy)}')

print(f'Training Loss: {np.min(loss)}')

print(f'Validation Accuracy: {np.max(val_accuracy)}')

print(f'Validation Loss: {np.min(val_loss)}')



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')