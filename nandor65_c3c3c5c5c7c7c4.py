

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# LOAD THE DATA

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



print (train.shape)

print (test.shape)

# PREPARE DATA FOR FEEDING YOUR CNN

from keras.utils.np_utils import to_categorical

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train / 255.0

X_test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)



print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization

from keras.initializers import he_normal

from keras.optimizers import Adam



nets = 15

model = [0] *nets



#Conv layer kernel sizes for the top 6 layers

ks1=3

ks2=3

ks3=5

ks4=5

ks5=7

ks6=7

init = he_normal(seed=82)



for j in range(nets):

    model[j] = Sequential()



    model[j].add(Conv2D(32, kernel_size=ks1, activation='relu', kernel_initializer = init, input_shape = (28, 28, 1)))

    model[j].add(BatchNormalization())

    #model[j].add(Dropout(0.4))

    model[j].add(Conv2D(32, kernel_size=ks2, activation='relu', kernel_initializer = init ))

    model[j].add(BatchNormalization())

    #model[j].add(Dropout(0.4))

    model[j].add(Conv2D(32, kernel_size=ks3, activation='relu', kernel_initializer = init ))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.4))



    model[j].add(Conv2D(64, kernel_size=ks4, activation='relu', kernel_initializer = init ))

    model[j].add(BatchNormalization())

    #model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size=ks5, activation='relu', kernel_initializer = init ))

    model[j].add(BatchNormalization())

    #model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size=ks6, activation='relu', kernel_initializer = init ))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(0.4))



    model[j].add(Conv2D(128, kernel_size=4, activation='relu', kernel_initializer = init ))

    model[j].add(BatchNormalization())

    model[j].add(Flatten())

    model[j].add(Dropout(0.4))

    model[j].add(Dense(10, activation='softmax'))



    optA = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model[j].compile(optimizer=optA, loss="categorical_crossentropy", metrics=["accuracy"])



model[0].summary()

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.10, width_shift_range=0.1, height_shift_range=0.1)
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.31623, min_delta=1e-4)

early_stopping = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=10, verbose=1, mode='max', restore_best_weights=True)



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



history = [0] * nets

epochs = 50

for j in range(nets):

    rs = 10*j + 1

    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, train_size = 0.9, random_state = rs)

    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64), epochs = epochs, steps_per_epoch = X_train2.shape[0]//64, validation_data = (X_val2,Y_val2), callbacks=[learning_rate_reduction, early_stopping], verbose=0)

    #pred = model[j].predict_classes(X_val2)

    #Y_val0 = np.argmax(Y_val2,axis=1) 

    maxpos = history[j].history['val_acc'].index(max(history[j].history['val_acc']))

    print("N{0:d}:: Max val_acc={1:.5f} at Epoch {2:d} Min_tr_loss={3:.5f} Min_val_loss={4:.5f}".format(j+1,max(history[j].history['val_acc']),maxpos+1, min(history[j].history['loss']), min(history[j].history['val_loss']) ))

results = np.zeros( (X_test.shape[0],10) ) 

for j in range(nets):

    results = results + model[j].predict(X_test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST_32C3C3C564C5C7C7128C4FC10_50e_LR10_RLROP_ES_BN_DA_DO40_TR90_N15_ADAM_xx.csv",index=False)



print(submission.shape)
