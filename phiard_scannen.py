import keras,os

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



trdata = ImageDataGenerator()

traindata = trdata.flow_from_directory(directory="../input/aksara-jawa/v3/train",target_size=(224,224))

tsdata = ImageDataGenerator()

testdata = tsdata.flow_from_directory(directory="../input/aksara-jawa/v3/test", target_size=(224,224))



model = Sequential()

model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))



model.add(Flatten())

model.add(Dense(units=4096,activation="relu"))

model.add(Dense(units=4096,activation="relu"))

model.add(Dense(units=20, activation="softmax"))



from keras.optimizers import Adam

opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])



model.summary()



from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=40,generator=traindata, validation_data= testdata, validation_steps=10,epochs=50,callbacks=[checkpoint,early])



import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"])

plt.plot(hist.history['val_accuracy'])

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title("model accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])

plt.show()

#PREDICTION



from keras.preprocessing import image

#img = image.load_img("../input/aksara-jawa/dataset/prediction/ra164.pred.png",target_size=(224,224))

img = image.load_img("../input/aksara-jawa/prediction/prediction/ba17.png",target_size=(224,224))

img = np.asarray(img)

plt.imshow(img)

img = np.expand_dims(img, axis=0)

from keras.models import load_model

saved_model = load_model("vgg16.h5")

output = saved_model.predict(img)



max = output[0][0]

pos = 0

for i in range(1, 19): 

    if output[0][i] > max: 

        max = output[0][i]

        pos = i

         

print(output)

print("Score: ")

print(max)

print("Aksara: ")

if (pos == 0) :

    print("ba")

elif (pos == 1) :

    print('ca')

elif (pos == 2) :

    print('da')

elif (pos == 3) :

    print('dha')

elif (pos == 4) :

    print('ga')

elif (pos == 5) :

    print('ha')

elif (pos == 6) :

    print('ja')

elif (pos == 7) :

    print('ka')

elif (pos == 8) :

    print('la')

elif (pos == 9) :

    print('ma')

elif (pos == 10) :

    print('na')

elif (pos == 11) :

    print('nga')

elif (pos == 12) :

    print('nya')

elif (pos == 13) :

    print('pa')

elif (pos == 14) :

    print('ra')

elif (pos == 15) :

    print('sa')

elif (pos == 16) :

    print('ta')

elif (pos == 17) :

    print('tha')

elif (pos == 18) :

    print('wa')

elif (pos == 19) :

    print('ya')
def file_size(fname):

        import os

        statinfo = os.stat(fname)

        return statinfo.st_size



print("File size in bytes of a plain file: ",file_size("../working/vgg16.h5"))
