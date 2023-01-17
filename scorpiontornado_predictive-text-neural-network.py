# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pwd
#!pip install --upgrade pip

# !pip install 'tensorflow==2.3'

#!pip install 'tensorflow==2.2'



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt 

import cv2 as cv



from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D

from keras import models

from keras.optimizers import Adam,RMSprop 

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



import pickle



import random



# import re

#import tensorflow as tf

# import numpy as np

# from matplotlib import pyplot as plt

#print("Tensorflow version " + tf.__version__)

# AUTO = tf.data.experimental.AUTOTUNE

# from kaggle_datasets import KaggleDatasets



%matplotlib inline
# # https://www.kaggle.com/docs/tpu

# #https://www.kaggle.com/mgornergoogle/five-flowers-with-keras-and-xception-on-tpu

# try: # detect TPUs

# #     tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() # TPU detection

# #     strategy = tf.distribute.TPUStrategy(tpu)



#     # detect and init the TPU

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)



#     # instantiate a distribution strategy

# #     tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

    

# except ValueError: # detect GPUs

#     strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines

#     #strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU

#     #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # for clusters of multi-GPU machines



# print("Number of accelerators: ", strategy.num_replicas_in_sync)
# GCS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_PATH"
mapping = {}

#mapping[0] = 5



with open("../input/emnist/emnist-balanced-mapping.txt") as f:

    for line in f.read().split("\n")[:-1]:

        split = line.split()

        mapping[int(split[0])] = int(split[1])

        # print(split)

        # print(int(split[0]))

        

print(mapping)

# print(chr(mapping[28])) # the value of 28 corresponds to the ASCII code 83, which corresponds to the character "S"
np.random.seed(1) # seed

df_train = pd.read_csv("../input/emnist/emnist-balanced-train.csv", names=["label"]+["pixel"+str(x) for x in range(784)]) # Loading Dataset

df_train = df_train.iloc[np.random.permutation(len(df_train))] # Random permutaion for dataset (seed is used to resample the same permutation every time)
df_train.head(5)
df_train.shape
sample_size = df_train.shape[0] # Training set size

validation_size = int(df_train.shape[0]*0.1) # Validation set size 



# train_x and train_y

train_x = np.asarray(df_train.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1]) # taking all columns expect column 0

train_y = np.asarray(df_train.iloc[:sample_size-validation_size,0]).reshape([sample_size-validation_size,1]) # taking column 0



# val_x and val_y

val_x = np.asarray(df_train.iloc[sample_size-validation_size:,1:]).reshape([validation_size,28,28,1])

val_y = np.asarray(df_train.iloc[sample_size-validation_size:,0]).reshape([validation_size,1])
train_x.shape,train_y.shape
# df_test = pd.read_csv("../input/digit-recognizer/test.csv")

# test_x = np.asarray(df_test.iloc[:,:]).reshape([-1,28,28,1])



df_test = pd.read_csv("../input/emnist/emnist-balanced-test.csv", names=["label"]+["pixel"+str(x) for x in range(784)])



test_x = np.asarray(df_test.iloc[:,1:]).reshape([-1,28,28,1])

#test_y = np.asarray(df_test.iloc[:,0]).reshape([-1,1])
# converting pixel values in range [0,1]

train_x = train_x/255

val_x = val_x/255

test_x = test_x/255
rows = 5 # defining no. of rows in figure

cols = 6 # defining no. of colums in figure



f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 



for i in range(rows*cols): 

    f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration

    plt.imshow(train_x[i].reshape([28,28]),cmap="Blues") 

    plt.axis("off")

    plt.title(str(train_y[i]), y=-0.15,color="green")

plt.savefig("digits.png")
rows = 5 # defining no. of rows in figure

cols = 6 # defining no. of colums in figure



f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 



for i in range(rows*cols): 

    f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration

    plt.imshow(train_x[i].reshape([28,28]),cmap="Blues") 

    plt.axis("off")

    plt.title(chr(mapping[int(train_y[i])]), y=-0.15,color="green")

plt.savefig("digits.png")
# # First attempt at getting one image

# f = plt.figure # defining a figure 

# print(plt.gcf().gca())

# plt.imshow(train_x[0].reshape([28,28]),cmap="Greys") 

# plt.axis("off")

# plt.title(str(train_y[0]), y=-0.15,color="green")

# plt.savefig("digits.png")





# # Attempt at OO

# fig,ax = plt.subplots() # defining a figure 

# #print(plt.gcf().gca())

# ax.imshow(train_x[0].reshape([28,28]),cmap="Greys") 

# ax.axis("off")

# ax.set_title(str(train_y[0]), y=-0.15,color="green")

# fig.savefig("digits.png")





# Least complicated

plt.imshow(train_x[0].reshape([28,28]),cmap="Greys") 

plt.axis("off")

plt.title(chr(mapping[int(train_y[0])]), y=-0.15,color="green")

plt.colorbar()

plt.savefig("digit.png")
cur_digit = train_x[0].reshape([28,28])

cur_digit = cur_digit.transpose()



plt.imshow(cur_digit,cmap="Greys") 

plt.axis("off")

plt.title(chr(mapping[int(train_y[0])]), y=-0.15,color="green")

plt.savefig("transposed_digit.png")
rows = 5 # defining no. of rows in figure

cols = 6 # defining no. of colums in figure



f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 



for i in range(rows*cols):

    cur_digit = train_x[i].reshape([28,28]).transpose()

    f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration

    plt.imshow(cur_digit,cmap="Greys") 

    plt.axis("off")

    plt.title(chr(mapping[int(train_y[i])]), y=-0.15,color="green")

plt.savefig("transposed_digits.png")
index = random.choice(range(len(train_x)))

cur_digit = train_x[index].reshape([28,28])

cur_digit = cur_digit.transpose()



plt.imshow(cur_digit,cmap="Greys") 

plt.axis("off")

plt.title(chr(mapping[int(train_y[index])]), y=-0.15,color="green")

plt.savefig("transposed_digit.png")
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

model = models.Sequential()
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

# Block 1

model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))

model.add(LeakyReLU())

model.add(Conv2D(32,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Block 2

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(Conv2D(64,3, padding  ="same"))

model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256,activation='relu'))

# model.add(Dense(32,activation='relu'))

# model.add(Dense(10,activation="sigmoid"))

model.add(Dense(64,activation='relu')) # Seems to work better without this layer; I'll do tests with and without it

model.add(Dense(47,activation="sigmoid"))



# Tried with 46, got the following error, so used 47 instead and it seems to work fine:

# InvalidArgumentError:  Received a label value of 46 which is outside the valid range of [0, 46).  Label values: 1 1 27 13 45 2 30 16 38 19 22 45 13 44 17 2 19 21 12 39 25 35 8 26 25 28 22 37 13 41 2 32 21 15 30 9 2 3 16 15 23 26 29 35 9 12 3 19 46 41 28 24 33 34 19 29 44 33 42 25 22 32 21 3 46 34 22 37 22 16 34 12 39 22 34 20 20 5 26 8 33 39 30 32 7 37 10 20 27 27 10 19 10 36 16 10 22 3 19 44 35 34 42 28 30 21 43 34 15 4 4 42 10 10 9 24 34 3 36 26 42 40 27 46 40 23 27 45 42 2 39 12 0 31 33 14 34 1 46 4 3 44 32 28 20 39 44 25 36 25 8 20 24 8 10 39 34 36 13 36 13 46 36 43 24 19 44 42 13 38 19 4 6 28 22 2 41 39 1 4 40 13 18 1 17 26 14 36 17 18 43 4 37 18 41 23 3 11 1 36 27 30 32 4 8 38 31 0 42 37 13 27 43 35 39 43 39 22 39 31 45 5 5 25 37 3 27 33 41 1 32 1 38 40 38 2 16 37 13 16 45 31 31 16 4 5 44 45 3 0 11 20 40 19 12 2

# 	 [[node sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits (defined at <ipython-input-90-0ad8c1b548b3>:4) ]] [Op:__inference_train_function_6144]



# Function call stack:

# train_function
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

initial_lr = 0.001

loss = "sparse_categorical_crossentropy"

model.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])

model.summary()
# epochs = 20

epochs = 20

batch_size = 256

# batch_size = 16 * strategy.num_replicas_in_sync

# https://stackoverflow.com/questions/61586981/valueerror-layer-sequential-20-expects-1-inputs-but-it-received-2-input-tensor

# validation_data has to be a tuple rather than a list (of Numpy arrays or tensors).

# history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=[val_x,val_y])

history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=(val_x,val_y))



# #model.save('path/to/location')

# # Calling `save('my_model')` creates a SavedModel folder `my_model`.

model.save("model_e20_1") # will work on CPU or GPU but not TPU



# # TPUs need this extra setting to save to local disk, otherwise, they can only save models to GCS (Google Cloud Storage).

# # The setting instructs Tensorflow to retrieve all parameters from the TPU then do the saving from the local VM, not the TPU.

# save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

# model.save('./model_e5_1', options=save_locally) # saving in Tensorflow's "saved model" format
# Defining Figure

f = plt.figure(figsize=(20,7))



#Adding Subplot 1 (For Accuracy)

f.add_subplot(121)



plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
val_p = np.argmax(model.predict(val_x),axis =1)



error = 0

# confusion_matrix = np.zeros([10,10])

confusion_matrix = np.zeros([47,47])

for i in range(val_x.shape[0]):

    confusion_matrix[val_y[i],val_p[i]] += 1

    if val_y[i]!=val_p[i]:

        error +=1

        

print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])

print("\nValidation set Shape :",val_p.shape[0])
f = plt.figure(figsize=(10,8.5))

f.add_subplot(111)



plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")

plt.colorbar()

plt.tick_params(size=5,color="white")

# plt.xticks(np.arange(0,10),np.arange(0,10))

# plt.yticks(np.arange(0,10),np.arange(0,10))

plt.xticks(np.arange(0,47),np.arange(0,47))

plt.yticks(np.arange(0,47),np.arange(0,47))



threshold = confusion_matrix.max()/2 



# for i in range(10):

#     for j in range(10):

#         plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

for i in range(47):

    for j in range(47):

        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black", fontsize=5)

        

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("Confusion_matrix1.png")

plt.show()



# for i in range(47):

#     print(str(i) + ": " + chr(mapping[int(train_y[i])]), end=",\t" if (i+1)%10 != 0 else "\n")

for i, key in enumerate(mapping):

    print(str(key) + ": " + chr(mapping[key]), end=",\t" if (i+1)%10 != 0 else "\n")
# instantiating the model in the strategy scope creates the model on the TPU

#with strategy.scope():

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(train_x)
lrr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5, min_lr=0.00001)
# epochs = 20

#history_2 = model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),steps_per_epoch=int(train_x.shape[0]/batch_size)+1,epochs=epochs,validation_data=[val_x,val_y],callbacks=[lrr])

epochs = 20

history_2 = model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),steps_per_epoch=int(train_x.shape[0]/batch_size)+1,epochs=epochs,validation_data=(val_x,val_y),callbacks=[lrr])

model.save("model_e20_2") # will work on CPU



# # TPUs need this extra setting to save to local disk, otherwise, they can only save models to GCS (Google Cloud Storage).

# # The setting instructs Tensorflow to retrieve all parameters from the TPU then do the saving from the local VM, not the TPU.

# save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

# model.save('./model_e5_2', options=save_locally) # saving in Tensorflow's "saved model" format
# Defining Figure

f = plt.figure(figsize=(20,7))

f.add_subplot(121)



#Adding Subplot 1 (For Accuracy)

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['accuracy']+history_2.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['val_accuracy']+history_2.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()





#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['loss']+history_2.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['val_loss']+history_2.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
val_p = np.argmax(model.predict(val_x),axis =1)



error = 0

#confusion_matrix = np.zeros([10,10])

confusion_matrix = np.zeros([47,47])

for i in range(val_x.shape[0]):

    confusion_matrix[val_y[i],val_p[i]] += 1

    if val_y[i]!=val_p[i]:

        error +=1

        

confusion_matrix,error,(error*100)/val_p.shape[0],100-(error*100)/val_p.shape[0],val_p.shape[0]



print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])

print("\nValidation set Shape :",val_p.shape[0])
f = plt.figure(figsize=(10,8.5))

f.add_subplot(111)



plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")

plt.colorbar()

plt.tick_params(size=5,color="white")

# plt.xticks(np.arange(0,10),np.arange(0,10))

# plt.yticks(np.arange(0,10),np.arange(0,10))

plt.xticks(np.arange(0,47),np.arange(0,47))

plt.yticks(np.arange(0,47),np.arange(0,47))



threshold = confusion_matrix.max()/2 



# for i in range(10):

#     for j in range(10):

#         plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

for i in range(47):

    for j in range(47):

        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black", fontsize=5)        

        

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("Confusion_matrix2.png")

plt.show()



# for i in range(47):

#     print(str(i) + ": " + chr(mapping[int(train_y[i])]), end=",\t" if (i+1)%10 != 0 else "\n")

for i, key in enumerate(mapping):

    print(str(key) + ": " + chr(mapping[key]), end=",\t" if (i+1)%10 != 0 else "\n")
rows = 4

cols = 9



f = plt.figure(figsize=(2*cols,2*rows))

sub_plot = 1

for i in range(val_x.shape[0]):

    if val_y[i]!=val_p[i] and sub_plot <= rows*cols:

        f.add_subplot(rows,cols,sub_plot) 

        sub_plot+=1

        plt.imshow(val_x[i].reshape([28,28]).transpose(),cmap="Blues")

        plt.axis("off")

#         plt.title("T: "+str(val_y[i])+" P:"+str(val_p[i]), y=-0.15,color="Red")

        plt.title("T: "+chr(mapping[int(val_y[i])])+" P:"+chr(mapping[int(val_p[i])]), y=-0.15,color="Red")

chr(mapping[int(train_y[i])])

plt.savefig("error_plots.png")

plt.show()
test_y = np.argmax(model.predict(test_x),axis =1)
rows = 5

cols = 10



f = plt.figure(figsize=(2*cols,2*rows))



for i in range(rows*cols):

    f.add_subplot(rows,cols,i+1)

    plt.imshow(test_x[i].reshape([28,28]).transpose(),cmap="Blues")

    plt.axis("off")

    plt.title(chr(mapping[int(test_y[i])]))
df_submission = pd.DataFrame([df_test.index+1,test_y],["ImageId","Label"]).transpose()

df_submission.to_csv("submission.csv",index=False)
mapping_letters = {}



with open("../input/emnist/emnist-letters-mapping.txt") as f:

    for line in f.read().split("\n")[:-1]:

        split = line.split()

        mapping_letters[int(split[0])] = int(split[1])

        # print(split)

        # print(int(split[0]))

        

print(mapping_letters)

# print(chr(mapping_letters[28])) # the value of 28 corresponds to the ASCII code 83, which corresponds to the character "S"

# actually not for this, but I'm too lazy to change it
np.random.seed(1) # seed

df_train_letters = pd.read_csv("../input/emnist/emnist-letters-train.csv", names=["label"]+["pixel"+str(x) for x in range(784)]) # Loading Dataset

df_train_letters = df_train_letters.iloc[np.random.permutation(len(df_train_letters))] # Random permutaion for dataset (seed is used to resample the same permutation every time)
sample_size_letters = df_train_letters.shape[0] # Training set size

validation_size_letters = int(df_train_letters.shape[0]*0.1) # Validation set size 



# train_x and train_y

train_x_letters = np.asarray(df_train_letters.iloc[:sample_size_letters-validation_size_letters,1:]).reshape([sample_size_letters-validation_size_letters,28,28,1]) # taking all columns expect column 0

train_y_letters = np.asarray(df_train_letters.iloc[:sample_size_letters-validation_size_letters,0]).reshape([sample_size_letters-validation_size_letters,1]) # taking column 0



# val_x and val_y

val_x_letters = np.asarray(df_train_letters.iloc[sample_size_letters-validation_size_letters:,1:]).reshape([validation_size_letters,28,28,1])

val_y_letters = np.asarray(df_train_letters.iloc[sample_size_letters-validation_size_letters:,0]).reshape([validation_size_letters,1])
df_test_letters = pd.read_csv("../input/emnist/emnist-letters-test.csv", names=["label"]+["pixel"+str(x) for x in range(784)])



test_x_letters = np.asarray(df_test_letters.iloc[:,1:]).reshape([-1,28,28,1])

#test_y_letters = np.asarray(df_test_letters.iloc[:,0]).reshape([-1,1])
# converting pixel values in range [0,1]

train_x_letters = train_x_letters/255

val_x_letters = val_x_letters/255

test_x_letters = test_x_letters/255
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

model_letters = models.Sequential()
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

# Block 1

model_letters.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))

model_letters.add(LeakyReLU())

model_letters.add(Conv2D(32,3, padding  ="same"))

model_letters.add(LeakyReLU())

model_letters.add(MaxPool2D(pool_size=(2,2)))

model_letters.add(Dropout(0.25))



# Block 2

model_letters.add(Conv2D(64,3, padding  ="same"))

model_letters.add(LeakyReLU())

model_letters.add(Conv2D(64,3, padding  ="same"))

model_letters.add(LeakyReLU())

model_letters.add(MaxPool2D(pool_size=(2,2)))

model_letters.add(Dropout(0.25))



model_letters.add(Flatten())



model_letters.add(Dense(256,activation='relu'))

# model.add(Dense(32,activation='relu'))

# model.add(Dense(10,activation="sigmoid"))

model_letters.add(Dense(64,activation='relu')) # Seems to work better without this layer; I'll do tests with and without it

model_letters.add(Dense(27,activation="sigmoid"))



# Tried with 26, got the following error, so used 27 instead and it seems to work fine:

# InvalidArgumentError:  Received a label value of 26 which is outside the valid range of [0, 26).  Label values: 9 3 11 1 2 24 9 3 19 4 7 14 24 20 23 8 14 8 23 26 21 15 6 25 10 16 22 7 22 24 6 12 17 19 20 6 12 14 4 11 26 10 7 7 11 8 9 21 8 3 3 14 22 6 5 20 22 1 26 20 5 14 9 18 25 12 16 26 16 2 2 16 6 13 22 5 25 14 7 18 2 12 5 10 26 3 22 9 5 8 11 10 25 25 8 19 19 18 16 11 21 18 17 14 4 15 11 5 24 19 1 12 18 24 26 14 25 7 11 18 11 12 18 13 17 19 20 16 7 7 2 16 13 22 8 17 8 24 1 26 26 26 17 11 14 6 26 18 15 16 6 11 19 14 19 4 24 9 25 3 16 11 25 13 23 21 8 7 20 6 5 18 26 2 21 17 26 22 6 7 7 20 5 22 21 10 26 5 15 21 13 10 20 5 4 8 7 26 7 4 7 6 25 17 13 24 13 9 5 15 21 8 10 11 11 20 11 3 1 10 9 26 16 2 16 18 13 23 4 10 23 14 4 16 8 16 9 20 1 3 18 10 4 14 19 23 21 21 17 19 26 4 13 25 20 21

# 	 [[node sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits (defined at <ipython-input-14-a0b98d3f7245>:8) ]] [Op:__inference_train_function_1164]



# Function call stack:

# train_function
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

initial_lr = 0.001

loss = "sparse_categorical_crossentropy"

model_letters.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])

model_letters.summary()
# epochs = 20

epochs = 20

batch_size = 256

# batch_size = 16 * strategy.num_replicas_in_sync

# https://stackoverflow.com/questions/61586981/valueerror-layer-sequential-20-expects-1-inputs-but-it-received-2-input-tensor

# validation_data has to be a tuple rather than a list (of Numpy arrays or tensors).

# history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=[val_x,val_y])

history_1_letters = model_letters.fit(train_x_letters,train_y_letters,batch_size=batch_size,epochs=epochs,validation_data=(val_x_letters,val_y_letters))



# #model.save('path/to/location')

# # Calling `save('my_model')` creates a SavedModel folder `my_model`.

model_letters.save("model_letters_e20_1") # will work on CPU or GPU but not TPU



# # TPUs need this extra setting to save to local disk, otherwise, they can only save models to GCS (Google Cloud Storage).

# # The setting instructs Tensorflow to retrieve all parameters from the TPU then do the saving from the local VM, not the TPU.

# save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

# model.save('./model_e5_1', options=save_locally) # saving in Tensorflow's "saved model" format
# Defining Figure

f = plt.figure(figsize=(20,7))



#Adding Subplot 1 (For Accuracy)

f.add_subplot(121)



plt.plot(history_1_letters.epoch,history_1_letters.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1_letters.epoch,history_1_letters.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1_letters.epoch,history_1_letters.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1_letters.epoch,history_1_letters.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
val_p_letters = np.argmax(model_letters.predict(val_x_letters),axis =1)



error = 0

# confusion_matrix = np.zeros([10,10])

confusion_matrix = np.zeros([27,27])

for i in range(val_x_letters.shape[0]):

    confusion_matrix[val_y_letters[i],val_p_letters[i]] += 1

    if val_y_letters[i]!=val_p_letters[i]:

        error +=1

        

print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p_letters.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p_letters.shape[0])

print("\nValidation set Shape :",val_p_letters.shape[0])
f = plt.figure(figsize=(10,8.5))

f.add_subplot(111)



plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")

plt.colorbar()

plt.tick_params(size=5,color="white")

# plt.xticks(np.arange(0,10),np.arange(0,10))

# plt.yticks(np.arange(0,10),np.arange(0,10))

plt.xticks(np.arange(0,27),np.arange(0,27))

plt.yticks(np.arange(0,27),np.arange(0,27))



threshold = confusion_matrix.max()/2 



# for i in range(10):

#     for j in range(10):

#         plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

for i in range(27):

    for j in range(27):

        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black", fontsize=8)

        

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("Confusion_matrix1_letters.png")

plt.show()



# for i in range(27):

#     print(str(i) + ": " + chr(mapping_letters[int(train_y_letters[i])]), end=",\t" if (i+1)%10 != 0 else "\n")

for i, key in enumerate(mapping_letters):

    print(str(key) + ": " + chr(mapping_letters[key]), end=",\t" if (i+1)%10 != 0 else "\n")

# instantiating the model in the strategy scope creates the model on the TPU

#with strategy.scope():

datagen_letters = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen_letters.fit(train_x_letters)
lrr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5, min_lr=0.00001)
# epochs = 20

#history_2 = model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),steps_per_epoch=int(train_x.shape[0]/batch_size)+1,epochs=epochs,validation_data=[val_x,val_y],callbacks=[lrr])

epochs = 20

history_2_letters = model_letters.fit_generator(datagen_letters.flow(train_x_letters,train_y_letters, batch_size=batch_size),steps_per_epoch=int(train_x_letters.shape[0]/batch_size)+1,epochs=epochs,validation_data=(val_x_letters,val_y_letters),callbacks=[lrr])

model_letters.save("model_letters_e20_2") # will work on CPU



# # TPUs need this extra setting to save to local disk, otherwise, they can only save models to GCS (Google Cloud Storage).

# # The setting instructs Tensorflow to retrieve all parameters from the TPU then do the saving from the local VM, not the TPU.

# save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

# model.save('./model_e5_2', options=save_locally) # saving in Tensorflow's "saved model" format
# Defining Figure

f = plt.figure(figsize=(20,7))

f.add_subplot(121)



#Adding Subplot 1 (For Accuracy)

plt.plot(history_1_letters.epoch+list(np.asarray(history_2_letters.epoch) + len(history_1_letters.epoch)),history_1_letters.history['accuracy']+history_2_letters.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1_letters.epoch+list(np.asarray(history_2_letters.epoch) + len(history_1_letters.epoch)),history_1_letters.history['val_accuracy']+history_2_letters.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()





#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1_letters.epoch+list(np.asarray(history_2_letters.epoch) + len(history_1_letters.epoch)),history_1_letters.history['loss']+history_2_letters.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1_letters.epoch+list(np.asarray(history_2_letters.epoch) + len(history_1_letters.epoch)),history_1_letters.history['val_loss']+history_2_letters.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
val_p_letters = np.argmax(model_letters.predict(val_x_letters),axis =1)



error = 0

#confusion_matrix = np.zeros([10,10])

confusion_matrix = np.zeros([27,27])

for i in range(val_x_letters.shape[0]):

    confusion_matrix[val_y_letters[i],val_p_letters[i]] += 1

    if val_y_letters[i]!=val_p_letters[i]:

        error +=1

        

confusion_matrix,error,(error*100)/val_p_letters.shape[0],100-(error*100)/val_p_letters.shape[0],val_p_letters.shape[0]



print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p_letters.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p_letters.shape[0])

print("\nValidation set Shape :",val_p_letters.shape[0])
f = plt.figure(figsize=(10,8.5))

f.add_subplot(111)



plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")

plt.colorbar()

plt.tick_params(size=5,color="white")

# plt.xticks(np.arange(0,10),np.arange(0,10))

# plt.yticks(np.arange(0,10),np.arange(0,10))

plt.xticks(np.arange(0,27),np.arange(0,27))

plt.yticks(np.arange(0,27),np.arange(0,27))



threshold = confusion_matrix.max()/2 



# for i in range(10):

#     for j in range(10):

#         plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

for i in range(27):

    for j in range(27):

        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")        

        

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("Confusion_matrix2_letters.png")

plt.show()



# for i in range(27):

#     print(str(i) + ": " + chr(mapping_letters[int(train_y_letters[i])]), end=",\t" if (i+1)%10 != 0 else "\n")

for i, key in enumerate(mapping_letters):

    print(str(key) + ": " + chr(mapping_letters[key]), end=",\t" if (i+1)%10 != 0 else "\n")
rows = 4

cols = 9



f = plt.figure(figsize=(2*cols,2*rows))

sub_plot = 1

for i in range(val_x_letters.shape[0]):

    if val_y_letters[i]!=val_p_letters[i] and sub_plot <= rows*cols:

        f.add_subplot(rows,cols,sub_plot) 

        sub_plot+=1

        plt.imshow(val_x_letters[i].reshape([28,28]).transpose(),cmap="Blues")

        plt.axis("off")

#         plt.title("T: "+str(val_y[i])+" P:"+str(val_p[i]), y=-0.15,color="Red")

        plt.title("T: "+chr(mapping_letters[int(val_y_letters[i])])+" P:"+chr(mapping_letters[int(val_p_letters[i])]), y=-0.15,color="Red")

chr(mapping_letters[int(train_y_letters[i])])

plt.savefig("error_plots_letters.png")

plt.show()
test_y_letters = np.argmax(model_letters.predict(test_x_letters),axis =1)
rows = 5

cols = 10



f = plt.figure(figsize=(2*cols,2*rows))



for i in range(rows*cols):

    f.add_subplot(rows,cols,i+1)

    plt.imshow(test_x_letters[i].reshape([28,28]).transpose(),cmap="Blues")

    plt.axis("off")

    plt.title(chr(mapping_letters[int(test_y_letters[i])]))
df_submission_letters = pd.DataFrame([df_test_letters.index+1,test_y_letters],["ImageId","Label"]).transpose()

df_submission_letters.to_csv("submission_letters.csv",index=False)
mapping_digits = {}



with open("../input/emnist/emnist-digits-mapping.txt") as f:

    for line in f.read().split("\n")[:-1]:

        split = line.split()

        mapping_digits[int(split[0])] = int(split[1])

        # print(split)

        # print(int(split[0]))

        

print(mapping_digits)

# print(chr(mapping_digits[28])) # the value of 28 corresponds to the ASCII code 83, which corresponds to the character "S"

# actually not for this, but I'm too lazy to change it
np.random.seed(1) # seed

df_train_digits = pd.read_csv("../input/emnist/emnist-digits-train.csv", names=["label"]+["pixel"+str(x) for x in range(784)]) # Loading Dataset

df_train_digits = df_train_digits.iloc[np.random.permutation(len(df_train_digits))] # Random permutaion for dataset (seed is used to resample the same permutation every time)
sample_size_digits = df_train_digits.shape[0] # Training set size

validation_size_digits = int(df_train_digits.shape[0]*0.1) # Validation set size 



# train_x and train_y

train_x_digits = np.asarray(df_train_digits.iloc[:sample_size_digits-validation_size_digits,1:]).reshape([sample_size_digits-validation_size_digits,28,28,1]) # taking all columns expect column 0

train_y_digits = np.asarray(df_train_digits.iloc[:sample_size_digits-validation_size_digits,0]).reshape([sample_size_digits-validation_size_digits,1]) # taking column 0



# val_x and val_y

val_x_digits = np.asarray(df_train_digits.iloc[sample_size_digits-validation_size_digits:,1:]).reshape([validation_size_digits,28,28,1])

val_y_digits = np.asarray(df_train_digits.iloc[sample_size_digits-validation_size_digits:,0]).reshape([validation_size_digits,1])
df_test_digits = pd.read_csv("../input/emnist/emnist-digits-test.csv", names=["label"]+["pixel"+str(x) for x in range(784)])



test_x_digits = np.asarray(df_test_digits.iloc[:,1:]).reshape([-1,28,28,1])

#test_y_digits = np.asarray(df_test_digits.iloc[:,0]).reshape([-1,1])
# converting pixel values in range [0,1]

train_x_digits = train_x_digits/255

val_x_digits = val_x_digits/255

test_x_digits = test_x_digits/255
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

model_digits = models.Sequential()
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

# Block 1

model_digits.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))

model_digits.add(LeakyReLU())

model_digits.add(Conv2D(32,3, padding  ="same"))

model_digits.add(LeakyReLU())

model_digits.add(MaxPool2D(pool_size=(2,2)))

model_digits.add(Dropout(0.25))



# Block 2

model_digits.add(Conv2D(64,3, padding  ="same"))

model_digits.add(LeakyReLU())

model_digits.add(Conv2D(64,3, padding  ="same"))

model_digits.add(LeakyReLU())

model_digits.add(MaxPool2D(pool_size=(2,2)))

model_digits.add(Dropout(0.25))



model_digits.add(Flatten())



model_digits.add(Dense(256,activation='relu'))

# model.add(Dense(32,activation='relu'))

# model.add(Dense(10,activation="sigmoid"))

model_digits.add(Dense(32,activation='relu')) # Seems to work better without this layer; I'll do tests with and without it

model_digits.add(Dense(10,activation="sigmoid"))



# Tried with 46, got the following error, so used 47 instead and it seems to work fine:

# InvalidArgumentError:  Received a label value of 46 which is outside the valid range of [0, 46).  Label values: 1 1 27 13 45 2 30 16 38 19 22 45 13 44 17 2 19 21 12 39 25 35 8 26 25 28 22 37 13 41 2 32 21 15 30 9 2 3 16 15 23 26 29 35 9 12 3 19 46 41 28 24 33 34 19 29 44 33 42 25 22 32 21 3 46 34 22 37 22 16 34 12 39 22 34 20 20 5 26 8 33 39 30 32 7 37 10 20 27 27 10 19 10 36 16 10 22 3 19 44 35 34 42 28 30 21 43 34 15 4 4 42 10 10 9 24 34 3 36 26 42 40 27 46 40 23 27 45 42 2 39 12 0 31 33 14 34 1 46 4 3 44 32 28 20 39 44 25 36 25 8 20 24 8 10 39 34 36 13 36 13 46 36 43 24 19 44 42 13 38 19 4 6 28 22 2 41 39 1 4 40 13 18 1 17 26 14 36 17 18 43 4 37 18 41 23 3 11 1 36 27 30 32 4 8 38 31 0 42 37 13 27 43 35 39 43 39 22 39 31 45 5 5 25 37 3 27 33 41 1 32 1 38 40 38 2 16 37 13 16 45 31 31 16 4 5 44 45 3 0 11 20 40 19 12 2

# 	 [[node sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits (defined at <ipython-input-90-0ad8c1b548b3>:4) ]] [Op:__inference_train_function_6144]



# Function call stack:

# train_function
# instantiating the model in the strategy scope creates the model on the TPU

# with strategy.scope():

initial_lr = 0.001

loss = "sparse_categorical_crossentropy"

model_digits.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])

model_digits.summary()
# epochs = 20

epochs = 20

batch_size = 256

# batch_size = 16 * strategy.num_replicas_in_sync

# https://stackoverflow.com/questions/61586981/valueerror-layer-sequential-20-expects-1-inputs-but-it-received-2-input-tensor

# validation_data has to be a tuple rather than a list (of Numpy arrays or tensors).

# history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=[val_x,val_y])

history_1_digits = model_digits.fit(train_x_digits,train_y_digits,batch_size=batch_size,epochs=epochs,validation_data=(val_x_digits,val_y_digits))



# #model.save('path/to/location')

# # Calling `save('my_model')` creates a SavedModel folder `my_model`.

model_digits.save("model_digits_e20_1") # will work on CPU or GPU but not TPU



# # TPUs need this extra setting to save to local disk, otherwise, they can only save models to GCS (Google Cloud Storage).

# # The setting instructs Tensorflow to retrieve all parameters from the TPU then do the saving from the local VM, not the TPU.

# save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

# model.save('./model_e5_1', options=save_locally) # saving in Tensorflow's "saved model" format
# Defining Figure

f = plt.figure(figsize=(20,7))



#Adding Subplot 1 (For Accuracy)

f.add_subplot(121)



plt.plot(history_1_digits.epoch,history_1_digits.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1_digits.epoch,history_1_digits.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1_digits.epoch,history_1_digits.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1_digits.epoch,history_1_digits.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
val_p_digits = np.argmax(model_digits.predict(val_x_digits),axis =1)



error = 0

# confusion_matrix = np.zeros([10,10])

confusion_matrix = np.zeros([10,10])

for i in range(val_x_digits.shape[0]):

    confusion_matrix[val_y_digits[i],val_p_digits[i]] += 1

    if val_y_digits[i]!=val_p_digits[i]:

        error +=1

        

print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p_digits.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p_digits.shape[0])

print("\nValidation set Shape :",val_p_digits.shape[0])
f = plt.figure(figsize=(10,8.5))

f.add_subplot(111)



plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")

plt.colorbar()

plt.tick_params(size=5,color="white")

# plt.xticks(np.arange(0,10),np.arange(0,10))

# plt.yticks(np.arange(0,10),np.arange(0,10))

plt.xticks(np.arange(0,10),np.arange(0,10))

plt.yticks(np.arange(0,10),np.arange(0,10))



threshold = confusion_matrix.max()/2 



# for i in range(10):

#     for j in range(10):

#         plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

for i in range(10):

    for j in range(10):

        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

        

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("Confusion_matrix1_digits.png")

plt.show()



# for i in range(47):

#     print(str(i) + ": " + chr(mapping_digits[int(train_y_digits[i])]), end=",\t" if (i+1)%10 != 0 else "\n")

for i, key in enumerate(mapping_digits):

    print(str(key) + ": " + chr(mapping_digits[key]), end=",\t" if (i+1)%10 != 0 else "\n")
# instantiating the model in the strategy scope creates the model on the TPU

#with strategy.scope():

datagen_digits = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(train_x_digits)
lrr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5, min_lr=0.00001)
# epochs = 20

#history_2 = model.fit_generator(datagen.flow(train_x,train_y, batch_size=batch_size),steps_per_epoch=int(train_x.shape[0]/batch_size)+1,epochs=epochs,validation_data=[val_x,val_y],callbacks=[lrr])

epochs = 20

history_2_digits = model_digits.fit_generator(datagen_digits.flow(train_x_digits,train_y_digits, batch_size=batch_size),steps_per_epoch=int(train_x_digits.shape[0]/batch_size)+1,epochs=epochs,validation_data=(val_x_digits,val_y_digits),callbacks=[lrr])

model_digits.save("model_digits_e20_2") # will work on CPU



# # TPUs need this extra setting to save to local disk, otherwise, they can only save models to GCS (Google Cloud Storage).

# # The setting instructs Tensorflow to retrieve all parameters from the TPU then do the saving from the local VM, not the TPU.

# save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

# model.save('./model_e5_2', options=save_locally) # saving in Tensorflow's "saved model" format
# Defining Figure

f = plt.figure(figsize=(20,7))

f.add_subplot(121)



#Adding Subplot 1 (For Accuracy)

plt.plot(history_1_digits.epoch+list(np.asarray(history_2_digits.epoch) + len(history_1_digits.epoch)),history_1_digits.history['accuracy']+history_2_digits.history['accuracy'],label = "accuracy") # Accuracy curve for training set

plt.plot(history_1_digits.epoch+list(np.asarray(history_2_digits.epoch) + len(history_1_digits.epoch)),history_1_digits.history['val_accuracy']+history_2_digits.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set



plt.title("Accuracy Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Accuracy",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()





#Adding Subplot 1 (For Loss)

f.add_subplot(122)



plt.plot(history_1_digits.epoch+list(np.asarray(history_2_digits.epoch) + len(history_1_digits.epoch)),history_1_digits.history['loss']+history_2_digits.history['loss'],label="loss") # Loss curve for training set

plt.plot(history_1_digits.epoch+list(np.asarray(history_2_digits.epoch) + len(history_1_digits.epoch)),history_1_digits.history['val_loss']+history_2_digits.history['val_loss'],label="val_loss") # Loss curve for validation set



plt.title("Loss Curve",fontsize=18)

plt.xlabel("Epochs",fontsize=15)

plt.ylabel("Loss",fontsize=15)

plt.grid(alpha=0.3)

plt.legend()



plt.show()
val_p_digits = np.argmax(model_digits.predict(val_x_digits),axis =1)



error = 0

#confusion_matrix = np.zeros([10,10])

confusion_matrix = np.zeros([10,10])

for i in range(val_x_digits.shape[0]):

    confusion_matrix[val_y_digits[i],val_p_digits[i]] += 1

    if val_y_digits[i]!=val_p_digits[i]:

        error +=1

        

confusion_matrix,error,(error*100)/val_p_digits.shape[0],100-(error*100)/val_p_digits.shape[0],val_p_digits.shape[0]



print("Confusion Matrix: \n\n" ,confusion_matrix)

print("\nErrors in validation set: " ,error)

print("\nError Persentage : " ,(error*100)/val_p_digits.shape[0])

print("\nAccuracy : " ,100-(error*100)/val_p_digits.shape[0])

print("\nValidation set Shape :",val_p_digits.shape[0])
f = plt.figure(figsize=(10,8.5))

f.add_subplot(111)



plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")

plt.colorbar()

plt.tick_params(size=5,color="white")

# plt.xticks(np.arange(0,10),np.arange(0,10))

# plt.yticks(np.arange(0,10),np.arange(0,10))

plt.xticks(np.arange(0,10),np.arange(0,10))

plt.yticks(np.arange(0,10),np.arange(0,10))



threshold = confusion_matrix.max()/2 



# for i in range(10):

#     for j in range(10):

#         plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")

for i in range(10):

    for j in range(10):

        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")        

        

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("Confusion_matrix2_digits.png")

plt.show()



# for i in range(47):

#     print(str(i) + ": " + chr(mapping_digits[int(train_y_digits[i])]), end=",\t" if (i+1)%10 != 0 else "\n")

for i, key in enumerate(mapping_digits):

    print(str(key) + ": " + chr(mapping_digits[key]), end=",\t" if (i+1)%10 != 0 else "\n")
rows = 4

cols = 9



f = plt.figure(figsize=(2*cols,2*rows))

sub_plot = 1

for i in range(val_x_digits.shape[0]):

    if val_y_digits[i]!=val_p_digits[i] and sub_plot <= rows*cols:

        f.add_subplot(rows,cols,sub_plot) 

        sub_plot+=1

        plt.imshow(val_x_digits[i].reshape([28,28]).transpose(),cmap="Blues")

        plt.axis("off")

#         plt.title("T: "+str(val_y[i])+" P:"+str(val_p[i]), y=-0.15,color="Red")

        plt.title("T: "+chr(mapping_digits[int(val_y_digits[i])])+" P:"+chr(mapping_digits[int(val_p_digits[i])]), y=-0.15,color="Red")

chr(mapping_digits[int(train_y_digits[i])])

plt.savefig("error_plots_digits.png")

plt.show()
test_y_digits = np.argmax(model_digits.predict(test_x_digits),axis =1)
rows = 5

cols = 10



f = plt.figure(figsize=(2*cols,2*rows))



for i in range(rows*cols):

    f.add_subplot(rows,cols,i+1)

    plt.imshow(test_x_digits[i].reshape([28,28]).transpose(),cmap="Blues")

    plt.axis("off")

    plt.title(chr(mapping_digits[int(test_y_digits[i])]))
df_submission_digits = pd.DataFrame([df_test_digits.index+1,test_y_digits],["ImageId","Label"]).transpose()

df_submission_digits.to_csv("submission_digits.csv",index=False)