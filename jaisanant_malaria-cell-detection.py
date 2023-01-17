import numpy as np

from sklearn.metrics import confusion_matrix,classification_report

import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

import os

import matplotlib.pyplot as plt

os.mkdir("../output/")

dataset_path = "../input/cell_images/cell_images/"

output_path = "../output/"

# Image augmentation

train_datagen = ImageDataGenerator(rescale=1./255,

                                   horizontal_flip=True,

                                   vertical_flip = True,

                                  validation_split = 0.2)



# training and validation generators

train_gen = train_datagen.flow_from_directory(directory = dataset_path,

                                              target_size = (100,100),

                                              class_mode = 'categorical',

                                              batch_size = 20,

                                              subset = 'training')



val_gen = train_datagen.flow_from_directory(directory = dataset_path,

                                              target_size = (100,100),

                                              class_mode = 'categorical',

                                              batch_size = 5,

                                              shuffle = False,

                                              subset = 'validation')



# class indices

print("Classes : " + str(train_gen.class_indices))



# sequential model

model = Sequential()



model.add(Conv2D(64,(3,3), input_shape = (100,100,3)))

model.add(Activation('relu'))

model.add(Conv2D(64,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(128,(3,3)))

model.add(Activation('relu'))

model.add(Conv2D(128,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(2048))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(516))

model.add(Activation('relu'))

model.add(Dropout(0.2))



model.add(Dense(2))

model.add(Activation('softmax'))



print(model.summary())



# checkpoint

checkpointer = keras.callbacks.ModelCheckpoint(filepath='../output/weights.hdf5', monitor='val_acc', 

                                               verbose=1, save_best_only=True)

# initiate RMSprop optimizer

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



# compile model

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



# fit the model

history = model.fit_generator(train_gen,

                              steps_per_epoch=len(train_gen),

                              epochs=60,

                              validation_data=val_gen, 

                              validation_steps=len(val_gen), 

                              workers = 4,

                              callbacks=[checkpointer],

                              verbose = 0

                             )

#training and validation accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')



plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
#load the saved model

load_model = keras.models.load_model("../output/weights.hdf5")



# evaluate on validation generator

evaluate = load_model.evaluate_generator(val_gen,

                           steps= len(val_gen),

                             workers = 4)

print("Loss : {0}, Accuracy :{1}".format(evaluate[0],evaluate[1]))



# predict on validation generator

predict = load_model.predict_generator(val_gen,

                                      steps = len(val_gen),

                                      workers = 4)



print("Prediction : " + str(predict))

print('.'*100)

# predict[i][0] = parasitized probability and predict[i][1] = uninfected probability



#convert to 1 and 0

y_pred = np.rint(predict)



# combining both classes

pred =[]

for i in range(len(y_pred)) :

    if predict[i][0] >= 0.5 :

        pred.append(0)

    else :

        pred.append(1)



# true labels

y_true = val_gen.classes

print("True labels : ")

print(y_true)

print('.'*100)



# confusion matrix

print("Confusin Matrix : ")

print(confusion_matrix(y_true, pred))

print('.'*100)



# classification report

print("Classification Report : ")

print(classification_report(y_true,pred))