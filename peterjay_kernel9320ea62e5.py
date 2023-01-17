# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

y_train=train['label']

X_train=train.drop(labels=['label'], axis=1)
y_train.value_counts()

#plt.hist(Y_train.value_counts(),) we use the seaborn plot for easier/faster visualisation

sns.countplot(y_train)
from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization

from keras.utils import to_categorical

from keras.optimizers import Adam 

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

#from sklearn.metrics import confusion_matrix
X_train=X_train.values.reshape(-1,28,28,1)/255.0

test=test.values.reshape(-1,28,28,1)/255.0
plt.imshow(X_train[0][:,:,0])
y_train=to_categorical(y_train) #onehotencoding

print(y_train[0])
random_state=69

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train, test_size=0.2, random_state=random_state)
# Initialising the CNN

classifier = Sequential()



# Convolution

classifier.add(Conv2D(64, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))



# Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



#Batch Normalization

classifier.add(BatchNormalization())



# Dropout (for reducing overfitting)

classifier.add(Dropout(0.2))



# Adding a second convolutional layer+MaxPooling and Batch Normalization

classifier.add(Conv2D(64, 3, 3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))





# Adding a third convolutional layer,MaxPooling and Batch Normalization

classifier.add(Conv2D(64, 3, 3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# adding second Dropout

classifier.add(Dropout(0.2))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(output_dim = 512, activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(output_dim=512,activation='relu'))

classifier.add(Dense(output_dim = 10, activation = 'softmax'))



optimizer_specs='Adam'

batch_size=20

epochs=20

classifier.compile(optimizer =optimizer_specs, loss = 'categorical_crossentropy', metrics = ['accuracy'])
datagen = ImageDataGenerator(

    featurewise_center=False,

    featurewise_std_normalization=False,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=False)

# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(X_train)

history1=classifier.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                    steps_per_epoch=len(X_train) / epochs,validation_data=(X_val,y_val), epochs=epochs)
"""history=classifier.fit(X_train, y_train,

              batch_size=batch_size,

              epochs=epochs,

              validation_data=(X_val, y_val),

              shuffle=True)

              """
"""

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

"""
# Plot training & validation accuracy values

plt.plot(history1.history['acc'])

plt.plot(history1.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history1.history['loss'])

plt.plot(history1.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
"""# Plot training & validation accuracy values

plt.plot(history2.history['acc'])

plt.plot(history2.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history2.history['loss'])

plt.plot(history2.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

"""
prediction=classifier.predict(test,verbose=1)
result=np.argmax(prediction, axis=1)

result
submission=pd.DataFrame({"ImageId":list(range(1,len(result)+1)),'Label':result})
result=np.argmax(prediction, axis=1)

submission=pd.DataFrame({"ImageId":list(range(1,len(result)+1)),'Label':result})

submission.to_csv("sample_submission.csv",index=False)