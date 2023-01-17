# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split

from keras import  backend as K

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# create the training & test sets, skipping the header row with [1:]

train = pd.read_csv("../input/digit-recognizer/train.csv")

print(train.shape)

# n'affiche que les premières valeurs de la matrice train

train.head()

#train
test = pd.read_csv("../input/digit-recognizer/test.csv")

print(test.shape)

test.head()
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')

X_test
y_train
#Convert train datset to (num_images, img_rows, img_cols) format 

X_train = X_train.reshape(X_train.shape[0], 28, 28)



for i in range(0, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i]);
#expand 1 more dimention as 1 for colour channel gray

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_train.shape
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_test.shape
mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px
from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)



num_classes = y_train.shape[1]

num_classes
# fix random seed for reproducibility

seed = 43

np.random.seed(seed)
from keras.models import  Sequential

from keras.layers.core import  Lambda , Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
model= Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)
from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),

 loss='categorical_crossentropy',

 metrics=['accuracy'])
from keras.preprocessing import image

gen = image.ImageDataGenerator()
from sklearn.model_selection import train_test_split

X = X_train

y = y_train

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches=gen.flow(X_val, y_val, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=2, 

                    validation_data=val_batches, validation_steps=val_batches.n)
def generate_submission(history):

    values=history.predict(X_test)

    indices = [i+1 for i, v in enumerate(values)]

    df = pd.DataFrame(list(zip(indices, values)), columns=['ImageId', 'Label'])

    return df
predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)