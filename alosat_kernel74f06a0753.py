# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D,LeakyReLU

#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import RMSprop,Nadam,Adadelta

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.regularizers import l2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tf.test.gpu_device_name()
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')
print(train)
y_train=train["label"]

x_train= train.drop(labels=["label"],axis=1)

g = sns.countplot(y_train)



y_train.value_counts()
x_train.isnull().any().describe()

test.isnull().any().describe()

test= test.drop(labels=["id"],axis=1)
num=6

number = train.iloc[num,1:].values.reshape(28,28)

print("Picture of "+ str(num) + "in Kannada style")

plt.imshow(number, cmap=plt.get_cmap('gray'))

plt.show()
x_train = x_train / 255.0

test = test / 255.0

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

g = plt.imshow(x_train[0][:,:,0])

y_train = to_categorical(y_train, num_classes = 10)

random_seed = 2



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))#  

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 5

batch_size = 86
datagen = ImageDataGenerator(rotation_range=15,

                             width_shift_range = 0.15,

                             height_shift_range = 0.15,

                             shear_range = 0.15,

                             zoom_range = 0.4,

                              horizontal_flip = False)
datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val),

                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
model.evaluate(x_val, y_val, verbose=2);

# CM:

y_predicted = model.predict(x_val)

y_grand_truth = y_val

y_predicted = np.argmax(y_predicted,axis=1)

y_grand_truth = np.argmax(y_grand_truth,axis=1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_grand_truth, y_predicted)
f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.1, linecolor="purple", ax=ax)

plt.xlabel("Predicted")

plt.ylabel("Grand Truth")

plt.show()
scores = np.zeros((10,3))

def calc_F1(num):

  TP = cm[num,num]

  FN = np.sum(cm[num,:])-cm[num,num]

  FP = np.sum(cm[:,num])-cm[num,num]

  precision = TP/(TP+FP)

  recall = TP/(TP+FN)

  F1_score = 2*(recall * precision) / (recall + precision)

  return precision, recall, F1_score

for i in range(10):

   precision, recall, F1_score = calc_F1(i)

   scores[i,:] = precision, recall, F1_score

scores_frame = pd.DataFrame(scores,columns=["Precision", "Recall", "F1 Score"], index=[list(range(0, 10))])
f, ax = plt.subplots(figsize = (4,6))

ax.set_title('Number Scores')

sns.heatmap(scores_frame, annot=True, fmt=".3f", linewidths=0.5, cmap="PuBu", cbar=True, ax=ax)

bottom, top = ax.get_ylim()

plt.ylabel("")

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
#fig, ax = plt.subplots(2,1)

#ax[0].plot(history.history['loss'], color='b', label="Training loss")

#ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

#legend = ax[0].legend(loc='best', shadow=True)



#ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

#ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

#legend = ax[1].legend(loc='best', shadow=True)
results = model.predict(test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)
#sub=model.predict(test)     ##making prediction

#sub=np.argmax(sub,axis=1) ##changing the prediction intro labels



#sample_sub['label']=sub

#sample_sub.to_csv('submission.csv',index=False)