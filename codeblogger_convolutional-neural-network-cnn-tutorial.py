# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train.csv")

train_df.head()
pd.DataFrame({

    'X': ['Train Shape','Different number of labels','Different number of labels (Sum)' ],

    'Y': [train_df.shape, train_df.label.unique(), len(train_df.label.unique())],

})
test_df = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test.csv")

print("Test Shape: ", test_df.shape)

test_df.head()
my_circle = plt.Circle( (0,0), 0.7, color='white')

plt.pie([len(train_df),len(test_df)], labels=["Train","Test"], colors=['green','skyblue'])

p = plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
# I synchronize the numbers it represents to a variable in the form of an array

Y_train = train_df['label']

Y_test = test_df['label']



X_train = train_df.drop(["label"],axis=1)

X_test = test_df.drop(["label"],axis=1)



del train_df['label']

del test_df['label']
plt.figure(figsize=(15,7))

g = sns.countplot(Y_train, palette="icefire")

plt.title("Number of digit classes")

Y_train.value_counts()
f, ax = plt.subplots(2,4)

f.set_size_inches(8,8)



k = 0

for i in range(2):

    for j in range(4):

        img = X_train.iloc[k].to_numpy()

        img = img.reshape((28,28))

        ax[i,j].set_xlabel(chr(Y_train[k] + 65))

        ax[i,j].imshow(img,cmap='gray')

        k += 1

    plt.tight_layout()
# normalize the data

X_train = X_train/255.0

X_test = X_test/255.0

print("X_train shape: ",X_train.shape)

print("X_test shape: ",X_test.shape)
# reshape

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

print("X_train shape: ",X_train.shape)

print("X_test shape: ",X_test.shape)
# label encoding

from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

Y_train = label_binrizer.fit_transform(Y_train)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=42)



print("x_train shape",X_train.shape)

print("x_test shape",X_val.shape)

print("y_train shape",Y_train.shape)

print("y_test shape",Y_val.shape)
plt.imshow(X_train[10][:,:,0],cmap="gray")

plt.show()
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



model.add(Conv2D(filters=75 , kernel_size=(3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2), strides = 2 , padding = 'same'))



model.add(Conv2D(filters=50, kernel_size=(3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(MaxPool2D(pool_size=(2,2), strides = 2 , padding = 'same'))



model.add(Conv2D(filters=25, kernel_size=(3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(MaxPool2D(pool_size=(2,2) , strides = 2 , padding = 'same'))



model.add(Flatten())



model.add(Dense(units = 512 , activation = 'relu'))

model.add(Dropout(0.2))



model.add(Dense(units = 24 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 25  # for better result increase the epochs

batch_size = 200
# data augmentation

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=15,  # randomly rotate images in the range 15 degrees

        zoom_range = 0.5, # Randomly zoom image 5%

        width_shift_range=0.15,  # randomly shift images horizontally 15%

        height_shift_range=0.15,  # randomly shift images vertically 15%

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
fig , ax = plt.subplots(1,2)

train_acc = history.history['accuracy']

train_loss = history.history['loss']

fig.set_size_inches(12,4)



ax[0].plot(history.history['accuracy'])

ax[0].plot(history.history['val_accuracy'])

ax[0].set_title('Training Accuracy vs Validation Accuracy')

ax[0].set_ylabel('Accuracy')

ax[0].set_xlabel('Epoch')

ax[0].legend(['Train', 'Validation'], loc='upper left')



ax[1].plot(history.history['loss'])

ax[1].plot(history.history['val_loss'])

ax[1].set_title('Training Loss vs Validation Loss')

ax[1].set_ylabel('Loss')

ax[1].set_xlabel('Epoch')

ax[1].legend(['Train', 'Validation'], loc='upper left')



plt.show()
import seaborn as sns

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(16, 12))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap=sns.cubehelix_palette(8),fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()