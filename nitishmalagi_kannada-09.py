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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau
train_df =pd.read_csv('../input/Kannada-MNIST/train.csv')

test_df =pd.read_csv('../input/Kannada-MNIST/test.csv')
print("Train set size", train_df.shape)

print("Test set size", test_df.shape)
train_df.head()
test_df.head()
test_df = test_df.drop('id', axis = 1)

test_df.head()
y = train_df.label.value_counts()

sns.barplot(y.index,y)

plt.title('Count Plot')

plt.show()
# setting target values

X_train = train_df.drop('label',axis=1)

y_train = train_df.label
# Normalize pixel values

X_train = X_train/255

test_df = test_df/255
# reshaping the dataset

X_train = X_train.values.reshape(-1,28,28,1)

test_df = test_df.values.reshape(-1,28,28,1)



print('The shape of train set now is', X_train.shape)

print('The shape of test set now is', test_df.shape)
# one-hot encoding the target values

y_train = to_categorical(y_train)
# splitting the train data into train and validation sets - 80/20

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 

                                                    random_state=42,test_size=0.20)
#printing sample images

plt.imshow(X_train[0][:,:,0])

plt.show()
# Data augmentation

datagen = ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))
model.summary()
optimizer=Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999)

model.compile(optimizer=optimizer,loss=['categorical_crossentropy'],metrics=['accuracy'])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, 

                                            factor=0.5, min_lr=0.00001)
# parameters

epochs = 5 

batch_size = 64
# Fitting the model

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])

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
# confusion matrix

y_pre_test=model.predict(X_test)

y_pre_test=np.argmax(y_pre_test,axis=1)

y_test=np.argmax(y_test,axis=1)



conf=confusion_matrix(y_test,y_pre_test)

conf=pd.DataFrame(conf,index=range(0,10),columns=range(0,10))



conf
# Making a Submission 

y_pred = model.predict(test_df)     #making prediction

y_pred = np.argmax(y_pred, axis=1)  #changing the prediction intro labels
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

sample_sub['label']=y_pred

sample_sub.to_csv('submission.csv',index=False)