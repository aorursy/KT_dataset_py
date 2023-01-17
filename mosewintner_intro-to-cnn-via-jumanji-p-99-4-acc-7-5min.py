import os

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



sns.set(style='white')



SEED = 323

VAL_FRAC = 0.1

EPOCHS = 20

BATCH_SIZE = 128
if os.path.isfile('../input/train.csv') and os.path.isfile('../input/test.csv'):

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    print('train.csv loaded: train({0[0]},{0[1]})'.format(train.shape))

    print('test.csv loaded: test({0[0]},{0[1]})'.format(test.shape))

else:

    print('Error: train.csv or test.csv not found in /input')

    

print('')

print(train.info())

print('')

print(train.isnull().any().describe())

print('')

print(train['label'].value_counts())
y_train = train['label']

X_train = train.iloc[:,1:]



print(y_train.shape, X_train.shape)

del train
X_train = X_train.values.reshape(-1,28,28,1)

X_test = test.values.reshape(-1,28,28,1)



print(X_train.shape, X_test.shape)
X_train = X_train.astype(np.float) # convert from int64 to float32

X_test = X_test.astype(np.float)

X_train = np.multiply(X_train, 1.0 / 255.0)

X_test = np.multiply(X_test, 1.0 / 255.0)
#CHECK: plot some images

plt.figure(figsize=(18,2))

for i in range(12):

    plt.subplot(2,12,1+i)

    plt.xticks(())

    plt.yticks(())

    plt.imshow(X_train[i].reshape(28,28),cmap=matplotlib.cm.binary)
idg = ImageDataGenerator(rotation_range=8.,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=np.pi/30., # 6 degrees

    zoom_range=0.1)



idg.fit(X_train)
y_train = to_categorical(y_train, num_classes = 10)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = VAL_FRAC)
model = Sequential()



model.add(Conv2D(16,(6,6), input_shape = (28,28,1), activation = "relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(4,4), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), activation = 'relu'))



model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))

model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',

             optimizer = Adam(),

             metrics = ['accuracy'])
lr = ReduceLROnPlateau(monitor='val_loss',

                       factor=0.5,

                       patience=2,

                       verbose=1,

                       epsilon=0.0001,

                       min_lr=1e-6)
fit = model.fit_generator(idg.flow(X_train,y_train,

                          batch_size=BATCH_SIZE),

                          epochs=EPOCHS,

                          validation_data=(X_val,y_val),

                          verbose=2,

                          steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

                          callbacks = [lr])
y_pred = model.predict(X_val)

# Convert predictions to one-hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation set labels to one-hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

conf = confusion_matrix(y_true, y_pred_classes)



plt.figure(figsize=(10,8))

sns.heatmap(pd.DataFrame(conf,range(10),range(10)), annot=True)
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)