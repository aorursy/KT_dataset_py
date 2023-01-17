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

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split



from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam





from sklearn.model_selection import train_test_split

from keras.layers.normalization import BatchNormalization
train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')



print(train.shape)

print(test.shape)
test.head(3)
train.head(3)
X = np.array(train.iloc[:, 1:])

y = to_categorical(np.array(train.iloc[:, 0]))
# y = train['label'].values

# train.drop(['label'], axis=1, inplace=True)
# X = train

# X.head(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Make out put Dic for Visualization



output_label_dic = {

0:"T-shirt/top",

1 :"Trouser",

2 :"Pullover",

3 :"Dress",

4 :"Coat",

5 :"Sandal",

6 :"Shirt",

7 :"Sneaker",

8 :"Bag",

9 :"Ankle boot" }



def nameOfLabel(digit):

    return output_label_dic[digit]

len(np.array(X_train[:1]))
def plt_dynamic(x, vy, ty,ax, colors=['b']):

# %matplotlib notebook

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()
f, ax = plt.subplots(1,2)

f.set_size_inches(10, 5)

ax[0].imshow(X_train[0].reshape(28, 28))

ax[1].imshow(X_train[2].reshape(28, 28))

plt.show()
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)





# X = np.array(data_train.iloc[:, 1:])

# y = to_categorical(np.array(data_train.iloc[:, 0]))



# #Here we split validation data to optimiza classifier during training

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)



#Test data

# X_test = np.array(data_test.iloc[:, 1:])

# y_test = to_categorical(np.array(data_test.iloc[:, 0]))







X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
num_classes = 10

epochs_ = 10

batch_size = 128
# Kearas Sequestional Model







model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss="categorical_crossentropy", optimizer="adam",

#               optimizer=optimizerArg,

              metrics=['accuracy'])

model_history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_, verbose=1, validation_data=(X_train, y_train))



score = model.evaluate(X_train, y_train, verbose=0)





print('Test loss:', score[0])

print('Test accuracy:', score[1])

train_loss_arr = model_history.history['val_loss']

test_loss_arr = model_history.history['loss']

model_train_loss = train_loss_arr[len(train_loss_arr) - 1]

model_test_loss = test_loss_arr[len(test_loss_arr) - 1]

%matplotlib inline

fig,layerArt_ax = plt.subplots(1,1)

layerArt_ax.set_xlabel('epoch') ; layerArt_ax.set_ylabel('Categorical Crossentropy Loss')

fig.suptitle('Train and test loss plot' , fontsize=14)

#Epoch num

layerArt_2_x = list(range(1, epochs_+1))



plt_dynamic(layerArt_2_x, train_loss_arr, test_loss_arr, layerArt_ax)


