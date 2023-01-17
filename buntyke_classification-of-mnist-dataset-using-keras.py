# import libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam ,RMSprop

from keras.layers import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Dropout



# fix random seed for reproducibility

seed = 43

np.random.seed(seed)



%matplotlib inline
# loading training and test datasets

test= pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")



# create training and test labels

y_train = train.iloc[:,0].values.astype('int32') 

X_train = (train.iloc[:,1:].values).astype('float32') 



X_test = test.values.astype('float32')
#Convert train datset to (num_images, img_rows, img_cols) format 

X_train = X_train.reshape(X_train.shape[0], 28, 28)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])    
#expand 1 more dimention as 1 for colour channel gray

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)



# standardize dataset

mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px



# convert to one-hot encoding

y_train = to_categorical(y_train)

num_classes = y_train.shape[1]
# no hidden layers

model= Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(350))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(100))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(num_classes, activation='softmax'))

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)



model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# partition to train and val

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)



# create an image generator and batches

gen = ImageDataGenerator()

batches = gen.flow(X_train, y_train, batch_size=16)

val_batches=gen.flow(X_val, y_val, batch_size=16)



# train the model and get performance data

history=model.fit_generator(batches, batches.n, nb_epoch=1, validation_data=val_batches, 

                            nb_val_samples=val_batches.n)

history_dict = history.history
epochs = range(1, len(loss_values)+1)



acc_values = history_dict['acc']

loss_values = history_dict['loss']

val_acc_values = history_dict['val_acc']

val_loss_values = history_dict['val_loss']



# visualize accuracies

plt.figure()

plt.plot(epochs, loss_values, 'bo')

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()



# visualize losses

plt.figure()   

plt.plot(epochs, acc_values, 'bo')

plt.plot(epochs, val_acc_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()
# apply predictions to model

predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})

submissions.to_csv("fcnn.csv", index=False, header=True)
# no hidden layers

model= Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))

model.add(Conv2D(10,(3,3)))

model.add(MaxPooling2D())

model.add(Conv2D(20,(3,3)))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(500))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(num_classes, activation='softmax'))

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)



model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# partition to train and val

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)



# create an image generator and batches

gen = ImageDataGenerator()

batches = gen.flow(X_train, y_train, batch_size=16)

val_batches=gen.flow(X_val, y_val, batch_size=16)



# train the model and get performance data

history=model.fit_generator(batches, batches.n, nb_epoch=1, validation_data=val_batches, 

                            nb_val_samples=val_batches.n)

history_dict = history.history
epochs = range(1, len(loss_values)+1)



acc_values = history_dict['acc']

loss_values = history_dict['loss']

val_acc_values = history_dict['val_acc']

val_loss_values = history_dict['val_loss']



# visualize accuracies

plt.figure()

plt.plot(epochs, loss_values, 'bo')

plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()



# visualize losses

plt.figure()   

plt.plot(epochs, acc_values, 'bo')

plt.plot(epochs, val_acc_values, 'b+')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()
# apply predictions to model

predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})

submissions.to_csv("cnn.csv", index=False, header=True)