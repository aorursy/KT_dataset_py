# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# ref taken from <https://www.kaggle.com/vikassingh1996/simple-cnn-modeling-kannanda-mnist>



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



'''Importing preprocessing libraries'''

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



'''Seaborn and Matplotlib Visualization'''

import matplotlib.pyplot as plt

import seaborn as sns





'''Importing tensorflow libraries'''

import tensorflow as tf 

print(tf.__version__)



from tensorflow.keras import layers, models



from keras.optimizers import RMSprop,Adam

from keras.utils.np_utils import to_categorical

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

from keras.models import Sequential

from keras import backend as K





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train =pd.read_csv(os.path.join(dirname,'train.csv'))

test =pd.read_csv(os.path.join(dirname,'test.csv'))

sample_submission =pd.read_csv(os.path.join(dirname,'sample_submission.csv'))
#display(np.unique(train)) 

display(train.head(1)) 

display(np.unique(train.head(1)))

display(train.shape) 

X_train=train.drop('label',axis=1)

Y_train=train.label

X_test = test.drop('id', axis = 1)

X_train = X_train / 255.0

X_test = X_test / 255.0
display(X_train.shape)

X_train =X_train.values.reshape(-1,28,28,1)

display(X_train.shape)

X_test=X_test.values.reshape(-1,28,28,1)

display(X_test.shape)
Y_train = to_categorical(Y_train,num_classes=10)

display(Y_train)
X_train,X_val,y_train,y_val=train_test_split(X_train,Y_train,random_state=42,test_size=0.10)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

from keras.layers import LeakyReLU
kernel_size_3 = (3,3)

kernel_size_5 = (5,5)

filters_32 = 32

filters_64 = 64

filters_128 = 128

filters_256 = 256



model = Sequential()

model.add(Conv2D(filters_64, kernel_size_3, activation='relu', input_shape=(28,28,1),padding='same' ))

model.add(Conv2D(filters_64, kernel_size_3, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.3))

#model.add(BatchNormalization())



          

model.add(Conv2D(filters_128, kernel_size_5, activation='relu',padding='same'))

model.add(Conv2D(filters_128, kernel_size_5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(MaxPool2D((2, 2)))

#model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(filters_256, kernel_size_5, activation='relu'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

#model.add(BatchNormalization())

model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.3))

          

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))
model.summary();
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
import seaborn as sns

import matplotlib.pyplot as plt

target = train.label

plt.figure(figsize=(15,5))

sns.countplot(target, color='crimson')

plt.title('The distribution of the digits in the dataset', weight='bold', fontsize='18')

plt.xticks(weight='bold', fontsize=16)

plt.show()
plt.figure(figsize=(15,8))

for i in range(60):

    plt.subplot(6,10,i+1)

    plt.imshow(X_train[i].reshape((28,28)),cmap='binary')

    plt.axis("off")

plt.show()
history = model.fit(X_train, y_train, batch_size = 256, epochs = 10, validation_data = (X_val, y_val), verbose = 2)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range = 10,

    horizontal_flip = False,

    zoom_range = 0.15)

datagen.fit(X_train)
BATCH_SIZE = 256

EPOCHS = 10
# training

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=BATCH_SIZE),

                              epochs = EPOCHS,

                              shuffle=True,

                              validation_data = (X_val,y_val),

                              verbose = 1,

                              steps_per_epoch=X_train.shape[0] // BATCH_SIZE)
epochs =10

fig,ax=plt.subplots(2,1)

fig.set

x=range(1,1+epochs)

ax[0].plot(x,history.history['loss'],color='red')

ax[0].plot(x,history.history['val_loss'],color='blue')



ax[1].plot(x,history.history['accuracy'],color='red')

ax[1].plot(x,history.history['val_accuracy'],color='blue')

ax[0].legend(['trainng loss','validation loss'])

ax[1].legend(['trainng acc','validation acc'])

plt.xlabel('Number of epochs')

plt.ylabel('accuracy')
y_pre_test=model.predict(X_val)

y_pre_test=np.argmax(y_pre_test,axis=1)

y_test=np.argmax(y_val,axis=1)
conf=confusion_matrix(y_test,y_pre_test)

conf=pd.DataFrame(conf,index=range(0,10),columns=range(0,10))
x=(y_pre_test-y_test!=0).tolist()

x=[i for i,l in enumerate(x) if l!=False]
fig,ax=plt.subplots(1,4,sharey=False,figsize=(15,15))



for i in range(4):

    ax[i].imshow(X_test[x[i]][:,:,0])

    ax[i].set_xlabel('Real {}, Predicted {}'.format(y_test[x[i]],y_pre_test[x[i]]))
'''confusion matrix'''

import seaborn as sns

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
'''predict results'''

results = model.predict(X_test)

'''select the indix with the maximum probability'''

results = np.argmax(results,axis = 1)
sample_sub =pd.read_csv(os.path.join(dirname,'sample_submission.csv'))

sample_sub['label'] = results

sample_sub.to_csv('submission.csv',index=False)