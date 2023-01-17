import numpy as np

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.convolutional import AveragePooling2D

from keras.layers import Activation

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils

from keras.models import load_model

from keras import optimizers

from sklearn.metrics import confusion_matrix

import pandas as pd

import seaborn as sns

import json

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



plt.style.use('ggplot')
from numpy import genfromtxt



train = genfromtxt("../input/train.csv",delimiter=',', skip_header=1)

test = genfromtxt("../input/test.csv",delimiter=',', skip_header=1)
X = train[:,1:]

y = train[:,0]



print("X_train shape: {}".format(X.shape))

print("y_train shape: {}".format(y.shape))

print("X_test shape: {}".format(test.shape))



print("Classes: {}".format(np.unique(y)))



num_classes = len(np.unique(y))

print("Number of classes: {}".format(num_classes))

X_reshaped = X.reshape((X.shape[0],28,28,1))

X_test_reshaped = test.reshape((test.shape[0],28,28,1))



# normalize inputs from 0-255 to 0-1

X_reshaped = X_reshaped/ 255

X_test_reshaped = X_test_reshaped / 255



# one hot encode outputs

y_oh = np_utils.to_categorical(y)



num_classes = y_oh.shape[1]

X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_oh, test_size=0.1, random_state=42)
#visualize first 9 images in training set



fig, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(4,4))

fig.subplots_adjust(hspace=0.5)

index = 0



for i in range(3):

    for j in range(3):

        ax[i, j].imshow(X_train[index,:,:,0], cmap=plt.get_cmap('gray'))

        ax[i, j].set_title("Label: {}".format(np.argmax(y[index])))

        index +=1



plt.show()
def create_lenet():

  # LeNet



  seed = 7

  np.random.seed(seed)





  # build the model

  # create Lenet model. Use same padding for 1st layer so that output size remains at 28 x 28 similar to Lenet output layer 1

  lenet_model = Sequential()

  lenet_model.add(Conv2D(6, kernel_size = (5, 5), strides = (1,1), input_shape=(28,28,1), padding = "same", activation='tanh'))

  lenet_model.add(AveragePooling2D(pool_size=(2, 2), strides = (2,2), padding = 'valid'))



  lenet_model.add(Conv2D(16, kernel_size = (5, 5), strides = (1,1), padding = "valid", activation='tanh'))

  lenet_model.add(AveragePooling2D(pool_size=(2, 2), strides = (2,2), padding = 'valid'))



  lenet_model.add(Conv2D(120, kernel_size = (5, 5), strides = (1,1), padding = "valid", activation='tanh'))



  lenet_model.add(Flatten())

  lenet_model.add(Dense(84, activation='tanh'))

  lenet_model.add(Dense(num_classes, activation='softmax'))

  

  return lenet_model

lenet_model= create_lenet()



# Compile model

lenet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fit the model

lenet_model_history = lenet_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, 

                                      batch_size=128, verbose=2, shuffle=True)

          
# evaluates model by comparing training and validation accuracy and losses



def model_evaluate(history):



  f, (ax1,ax2) = plt.subplots(2,1,sharex=True, figsize = (8,8))

  f.subplots_adjust(hspace=0.3)

  

  ax1.plot(history.history['acc'], 'r', linewidth=1)

  ax1.plot(history.history['val_acc'], 'b', linewidth=1)



  # Plot legend and use the best location automatically: loc = 0.

  ax1.legend(["Train Acc.", "Validation Acc."], loc = 0)

  ax1.set_title("Training/Validation Acc. per Epoch")

  ax1.set_ylabel("Accuracy")

     

  text = "Final Training Accuracy:{:.2f}%\nFinal Val. Accuracy:{:.2f}%".format(history.history['acc'][-1]*100,

                                                                            history.history['val_acc'][-1]*100)

  ax1.text(0.5, 0.5,text, transform=ax1.transAxes, fontsize=12)

  

  ax2.plot(history.history['loss'], 'r', linewidth=1)

  ax2.plot(history.history['val_loss'], 'b', linewidth=1)

  

  # Plot legend and use the best location automatically: loc = 0.

  ax2.legend(["Train loss", "Validation loss"], loc = 0)

  ax2.set_title("Training/Validation loss per Epoch")

  ax2.set_xlabel("Epoch")

  ax2.set_ylabel("Loss")

  

  plt.show()

  

  
#save trained model and model history

def save_files(model, model_name, model_history):

  model.save(model_name+".h5")

  files.download(model_name+".h5")

  

  with open(model_name+".json", 'w') as f:

    json.dump(model_history.history, f)

  files.download(model_name+".json")
model_evaluate(lenet_model_history)

#save_files(lenet_model, "lenet_model", lenet_model_history)
def create_lenet_dropout():



  seed = 7

  np.random.seed(seed)





  # build the model

  # create Lenet model. Use same padding for 1st layer so that output size remains at 28 x 28 similar to Lenet output layer 1

  lenet_model = Sequential()

  lenet_model.add(Conv2D(6, kernel_size = (5, 5), strides = (1,1), input_shape=(28,28,1), padding = "same", activation='relu'))

  lenet_model.add(AveragePooling2D(pool_size=(2, 2), strides = (2,2), padding = 'valid'))



  

  lenet_model.add(Conv2D(16, kernel_size = (5, 5), strides = (1,1), padding = "valid", activation='relu'))

  lenet_model.add(AveragePooling2D(pool_size=(2, 2), strides = (2,2), padding = 'valid'))

  lenet_model.add(Dropout(0.2))

  

  lenet_model.add(Conv2D(120, kernel_size = (5, 5), strides = (1,1), padding = "valid", activation='relu'))

  

  lenet_model.add(Flatten())

  lenet_model.add(Dropout(0.2))

  lenet_model.add(Dense(84, activation='relu'))

  lenet_model.add(Dense(num_classes, activation='softmax'))

  

  return lenet_model





lenet_dropout_model= create_lenet_dropout()



# Compile model

lenet_dropout_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])





# Fit the model

lenet_dropout_model_history = lenet_dropout_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=128, verbose=2, shuffle=True)

          

model_evaluate(lenet_dropout_model_history)

#save_files(lenet_dropout_model, "lenet_dropout_model", lenet_dropout_model_history)
lenet_model_dataAug= create_lenet()



# Compile model

lenet_model_dataAug.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# create image generator to create augmented data from X_train

gen = ImageDataGenerator(rotation_range=5, width_shift_range=0.05, shear_range=0.2,

                         height_shift_range=0.05, zoom_range=0.05)





train_generator = gen.flow(X_train, y_train, batch_size=128)



lenet_model_dataAug_history = lenet_model_dataAug.fit_generator(train_generator, steps_per_epoch=60000//128, epochs=30, validation_data=(X_val, y_val), verbose=2)
model_evaluate(lenet_model_dataAug_history)

#save_files(lenet_model_dataAug, "lenet_model_dataAug", lenet_model_dataAug_history)
def create_model_4Conv_3FC():

  seed = 7

  np.random.seed(seed)

  

  model = Sequential()

  model.add(Conv2D(12, kernel_size = (5, 5), strides = (1,1), input_shape=(28,28,1), padding="same", activation='relu'))

  model.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2)))

  model.add(BatchNormalization())



  model.add(Conv2D(48, kernel_size = (3, 3), strides = (1,1), padding = "same", activation='relu'))

  model.add(Conv2D(48, kernel_size = (3, 3), strides = (1,1), padding = "same", activation='relu'))

  model.add(Conv2D(32, kernel_size = (3, 3), strides = (1,1), padding = "same", activation='relu'))

  model.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2)))

  model.add(Dropout(0.4))



  model.add(Flatten())

  model.add(Dense(512, activation='relu'))



  model.add(Dropout(0.4))

  model.add(Dense(512, activation='relu'))



  model.add(Dense(num_classes, activation='softmax'))





  return model



model_4Conv_3FC = create_model_4Conv_3FC()



adam = optimizers.Adam(lr=1e-4)



model_4Conv_3FC.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])



# vanilla model with original training set gave lower validation accuracy

#model_4Conv_3FC_history = model_4Conv_3FC.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=128, verbose=2, shuffle=True)          



#with data augmentation  

model_4Conv_3FC_history = model_4Conv_3FC.fit_generator(train_generator, steps_per_epoch=60000//128, epochs=30, validation_data=(X_val, y_val), verbose=2)
#save_files(model_4Conv_3FC, "model_4Conv_3FC", model_4Conv_3FC_history)



model_evaluate(model_4Conv_3FC_history)
def create_model_6Conv_3FC():

  seed = 7

  np.random.seed(seed)

  

  model = Sequential()

  model.add(Conv2D(24, kernel_size = (3, 3), strides = (1,1), input_shape=(28,28,1), activation='relu', padding='same'))

  model.add(Conv2D(24, kernel_size = (3, 3), strides = (1,1), activation='relu', padding='same'))

  model.add(Conv2D(24, kernel_size = (3, 3), strides = (1,1), activation='relu', padding='same'))

  model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

  model.add(BatchNormalization())

  

  model.add(Conv2D(24, kernel_size = (3, 3), strides = (1,1), activation='relu', padding='same'))

  model.add(Conv2D(24, kernel_size = (3, 3), strides = (1,1), activation='relu', padding='same'))

  model.add(Conv2D(24, kernel_size = (3, 3), strides = (1,1), activation='relu', padding='same'))

  model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

  model.add(BatchNormalization())



  model.add(Flatten())

  model.add(Dropout(0.2))

  model.add(Dense(512, activation='relu'))

  model.add(Dropout(0.2))

  model.add(Dense(512, activation='relu'))

  model.add(Dense(num_classes, activation='softmax'))





  return model



model_6Conv_3FC = create_model_6Conv_3FC()



adam = optimizers.Adam(lr=1e-4)



model_6Conv_3FC.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



# vanilla model with original data set

#model_6Conv_3FC_history = model_6Conv_3FC.fit(X_train, y_train, validation_data=(X_val, y_val),

#                                                  epochs=30, batch_size=128, verbose=2, shuffle=True)



#data augmentation

model_6Conv_3FC_history = model_6Conv_3FC.fit_generator(train_generator, steps_per_epoch=60000//128, epochs=30, validation_data=(X_val, y_val), verbose=2)         
model_evaluate(model_6Conv_3FC_history)

#save_files(model_6Conv_3FC, "model_6Conv_3FC", model_6Conv_3FC_history)
index = ["lenet_model","lenet_dropout_model","lenet_model_dataAug",

          "model_4Conv_3FC", "model_6Conv_3FC"]

model_history = [lenet_model_history.history,lenet_dropout_model_history.history,

                 lenet_model_dataAug_history.history,

                 model_4Conv_3FC_history.history, 

                 model_6Conv_3FC_history.history]

models = [lenet_model,lenet_dropout_model,lenet_model_dataAug,

          model_4Conv_3FC, model_6Conv_3FC]



trainingAcc = []

valAcc = []

trainingLoss = []

valLoss = []

numParams = []



for model, hist in zip(models, model_history):

  trainingAcc.append(round(hist['acc'][-1],4))

  valAcc.append(round(hist['val_acc'][-1],4))

  trainingLoss.append(round(hist['loss'][-1],4))

  valLoss.append(round(hist['val_loss'][-1],4))

  numParams.append(model.count_params())



df = pd.DataFrame(list(zip(trainingAcc, valAcc, trainingLoss, valLoss, numParams)), 

               columns =['trainingAcc', 'valAcc', 'trainingLoss', 'valLoss', 'numParams'], 

                 index = index) 

  

df
predictions = model_4Conv_3FC.predict(X_test_reshaped)

predictions_classes = np.argmax(predictions, axis=1)

filename = "submission.csv"



predictions_df = pd.DataFrame({'ImageId': np.arange(1,len(predictions_classes)+1), 'Label': predictions_classes})

predictions_df.to_csv(filename, index=False)
files.download("submission.csv")
