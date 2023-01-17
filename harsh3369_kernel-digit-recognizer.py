# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
os.chdir("../input")

#Importing the dataset



df_train = pd.read_csv('train.csv')

df_test= pd.read_csv('test.csv')

# Checking the first few observation

df_train.head()
#Checking for duplicated values

df_train.duplicated().sum()
#df_train.corr()
#Checking the count of target variable

df_train.label.value_counts()
#Creating X_train, and y_train



y_train = df_train['label']

X_train = df_train.drop('label', axis = 1)
#checking the statistics of y_train

print(y_train.value_counts().sort_index())

print(y_train.describe())
#import the seaborn to check the distribution of target variable



import seaborn as sns



sns.distplot(y_train)
#Checking for null values in target variable



y_train.isna().sum()

X_train = X_train/255

df_test = df_test/255
from keras.utils import to_categorical



y_train = to_categorical(y_train)
# Checking the uniue vectors formed 

unique_rows = np.unique(y_train, axis=0)

unique_rows
def decode(datum):

    return np.argmax(datum)
# Checking the mapped values

for i in range(unique_rows.shape[0]):

    datum = unique_rows[i]

    print('index: %d' % i)

    print('encoded datum: %s' % datum)

    decoded_datum = decode(unique_rows[i])

    print('decoded datum: %s' % decoded_datum)

    print()
X_train = X_train.values.reshape(-1,28,28,1)

X_train
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size =0.1 , random_state = 42)
df_test = df_test.values.reshape(-1,28,28,1)

df_test
g = plt.imshow(X_train[0][:,:,0])
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

model = Sequential()

model.add(Conv2D(filters = 64, padding= "Same", kernel_size = (2,2), activation = 'relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(filters= 32, kernel_size = (2,2), activation= "relu"))

model.add(MaxPool2D(pool_size= (2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 16, kernel_size = (2,2), activation = "relu", padding = "Same" ))

model.add(MaxPool2D(pool_size= (2,2)))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.25))

model.add(Dense(128, activation= "relu"))

model.add(Dropout(0.33))

model.add(Dense(10, activation = "softmax"))
model.summary()
from keras.utils import plot_model

# plot graph

plot_model(model, to_file='/model_summary.png')
#Defining optimizer

from keras.optimizers import RMSprop

optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay =0.0 )
#compile the model

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
#Optimizing learning rate

from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',

                                           patience = 3,

                                           verbose = 1,

                                           factor = 0.5,

                                           min_lr = 0.00001)
#Agumenting the data 



from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(featurewise_center = False,

                            samplewise_center = False,

                            featurewise_std_normalization= False,

                            samplewise_std_normalization= False,

                            zca_whitening= False,

                            rotation_range = 10,

                            zoom_range = 0.1,

                            width_shift_range = 0.1,

                            height_shift_range = 0.1,

                            horizontal_flip= False,

                            vertical_flip= False)



datagen.fit(X_train)
batch_size = 75

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),

                             epochs = 35, validation_data = (X_val, y_val),

                             verbose = 1, steps_per_epoch = X_train.shape[0]//batch_size,

                             callbacks = [learning_rate_reduction])
#Plot the loss and accuracy curves for the training and validation set



fig, ax  = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color = 'b', label = 'Training loss')

ax[0].plot(history.history['val_loss'], color = 'r', label = 'Validation loss')

legend = ax[0].legend(loc ='best', shadow = True)



ax[1].plot(history.history['acc'], color = 'b', label = 'Training Accuracy')

ax[1].plot(history.history['val_acc'], color = 'r', label = 'Validation Accuracy')

legend = ax[1].legend(loc = 'best', shadow =True)
### Plotting the confusion matrix

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_val)

y_pred = np.argmax(y_pred, axis = 1)

y_true = np.argmax(y_val, axis = 1)

cm = confusion_matrix(y_true, y_pred)

# plot the confusion matrix

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cm, cmap= "YlGnBu", annot=True, fmt='', ax=ax)

results = model.predict(df_test)

results = np.argmax(results, axis = 1)

results = pd.Series(results, name = 'Label')
os.chdir("../working")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)