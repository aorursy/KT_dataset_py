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
# Importing the Required Library
import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,Activation,Conv2D,MaxPooling2D,Flatten, BatchNormalization, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv("../input/digit-recognizer/train.csv")
x_test = pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
# taking the independent feature and the dependent feature in xtrain and ytrain of the traning data
x_train = train.drop(columns = 'label')
y_train = train['label']
x_test.shape,x_train.shape,y_train.shape
sns.countplot(y_train)
#data processing
x_train = x_train.values.reshape(-1, int(np.sqrt(784)), int(np.sqrt(784)), 1)/255.0
x_test =  x_test.values.reshape(-1, int(np.sqrt(784)), int(np.sqrt(784)), 1)/255.0
x_test.shape,x_train.shape,y_train.shape
rows = 5
cols = 6
counter = 0
fig = plt.figure(figsize=(15,7))
for i in range(1, rows*cols+1):
    fig.add_subplot(rows, cols, i)
    plt.imshow(np.squeeze(x_train[counter + i-1]), cmap='gray')
    plt.title(y_train[counter + i-1], fontsize=16)
    plt.axis(False)
    fig.add_subplot
counter += rows*cols
y_train = to_categorical(y_train, num_classes=10)
x_test.shape,x_train.shape,y_train.shape
x_train_trim ,x_valid , y_train_trim, y_valid = train_test_split(x_train, y_train, test_size= 0.1 ,random_state = 1455)
print(f'Training Set size: {x_train_trim.shape[0]}')
print(f'Validation Set size: {x_valid.shape[0]}')
model = Sequential()#69
input_shape = (28,28,1)
model.add(Conv2D(32, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))
model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image("model.png")
#Optimizer & model compile
#optimizer_rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
adam = Adam(lr = .00001)
model.compile(optimizer=adam, 
              loss = 'categorical_crossentropy',  
              metrics = ['accuracy'])
# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=23,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.3, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train_trim)
def build_lrfn(lr_start=1e-4, lr_max=1e-3, 
               lr_min=0, lr_rampup_epochs=16, 
               lr_sustain_epochs=0, lr_exp_decay=.8):

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

lrfn = build_lrfn()
lr_schedule = LearningRateScheduler(lrfn, verbose=True)

#Usually monitor='val_accuracy' should be tracked here. Since the training set is smaller let keep it limited to accuracy
checkpoint = ModelCheckpoint(
    filepath='best_weights.hdf5',
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True)
train_history = model.fit_generator(datagen.flow(x_train_trim, y_train_trim, 
                                                 batch_size=512),
                                    epochs = 50,
                                    validation_data = (x_valid,y_valid),
                                    verbose = 2, 
                                    steps_per_epoch=x_train_trim.shape[0] //512,
                                    callbacks=[lr_schedule,checkpoint])
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
my_submission = pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred})
my_submission.to_csv('submission.csv', index=False)
