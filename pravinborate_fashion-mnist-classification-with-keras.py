# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

sns.set()
train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
train.head()
test.head()
train.shape
test.shape
training_array = np.array(train,dtype='float32')
testing_array = np.array(test,dtype = 'float32')
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
i = random.randint(1,60000)

plt.figure()

plt.imshow(training_array[i,1:].reshape(28,28))

plt.grid(False)

plt.show()

label = int(training_array[i,0])

print(f'The image is for : {class_names[label]}')
W_grid = 15

L_grid = 15



# subplot return the figure and axes object

# And by using axes object we can plot specific figure at various location

fig , axes = plt.subplots(L_grid,W_grid,figsize=(17,17))  

axes = axes.ravel()          #Flaten the 15 * 15 matrix into 255 array 





n_training = len(training_array)  #get the length of training dataset





for i in np.arange(0,L_grid*W_grid):

    

    index = np.random.randint(0,n_training)

    axes[i].imshow(training_array[index,1:].reshape(28,28))

    axes[i].set_title(class_names[int(training_array[index,0])],fontsize=8)

    axes[i].axis('off')

    

    

plt.subplots_adjust(hspace=0.4)
X_train = training_array[:,1:]/255

y_train = training_array[:,0]
X_test = testing_array[:,1:]/255

y_test = testing_array[:,0]
from sklearn.model_selection import train_test_split
X_train ,X_validate , y_train,y_validate = train_test_split(X_train,y_train,test_size = 0.2,random_state = 12345)
X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_test = X_test.reshape(X_test.shape[0],28,28,1)

X_validate = X_validate.reshape(X_validate.shape[0],28,28,1)
print(f'shape of X train : {X_train.shape}')

print(f'shape of X test : {X_test.shape}')

print(f'shape of X validate : {X_validate.shape}')
print(f'shape of y train : {y_train.shape}')

print(f'shape of y test : {y_test.shape}')

print(f'shape of y validate : {y_validate.shape}')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import MaxPool2D,Flatten,Dense

from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train,num_classes=10)

y_cat_test = to_categorical(y_test,num_classes=10)

y_cat_validate = to_categorical(y_validate,num_classes=10)
print(f'shape of y train : {y_cat_train.shape}')

print(f'shape of y test : {y_cat_test.shape}')

print(f'shape of y validate : {y_cat_validate.shape}')
model = Sequential()



# CONVOLUTIONAL LAYER

model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))

# POOLING LAYER

model.add(MaxPool2D(pool_size=(2, 2)))



# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER

model.add(Flatten())



# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)

model.add(Dense(128, activation='relu'))



model.add(Dropout(0.5))



# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES

model.add(Dense(10, activation='softmax'))



# https://keras.io/metrics/

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy']) # we can add in additional metrics https://keras.io/metrics/
model.summary()
early_stopping = EarlyStopping(monitor='val_loss',patience=1)
model.fit(X_train,

         y_cat_train,

         epochs=50,

          verbose=1,

         validation_data=(X_validate,y_cat_validate),

         callbacks=[early_stopping])
model_history = pd.DataFrame(model.history.history)
model_history
model_history[['accuracy','val_accuracy']].plot()
model_history[['loss','val_loss']].plot()
evalution = model.evaluate(X_test,y_cat_test)

print(f'Test Accuracy : {evalution[1]}')
predict_class = model.predict_classes(X_test)
predict_class.shape
i = random.randint(0,predict_class.shape[0])

print(class_names[predict_class[i]])

print(class_names[int(y_test[i])])
w_gird = 5

l_gird = 5



fig,axes = plt.subplots(l_gird,w_gird,figsize=(12,12))



axes = axes.ravel()



for i in np.arange(0,l_gird*w_gird):

    axes[i].imshow(X_test[i].reshape(28,28))

    axes[i].set_title(f'{i}.Predict Class : {class_names[predict_class[i]]} \n True Class : {class_names[int(y_test[i])]}')

    axes[i].axis('off')

    

plt.subplots_adjust(wspace=0.9,hspace=0.7)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predict_class)
fig = plt.figure(figsize=(12,12))

sns.heatmap(cm,annot=True,cmap='viridis',fmt='d')
from sklearn.metrics import classification_report



num_classes = 10

target_names = ['Class {}'.format(i) for i in range(num_classes)]



print(classification_report(y_test,predict_class,target_names=target_names))