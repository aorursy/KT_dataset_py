# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical



import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D,LeakyReLU

from tensorflow.keras.optimizers import RMSprop,Nadam,Adadelta

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.regularizers import l2



import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_train = pd.read_csv('../input/Kannada-MNIST/train.csv')

raw_test = pd.read_csv('../input/Kannada-MNIST/test.csv')
num = raw_train.label.value_counts()

sns.barplot(num.index,num)

numbers = num.index.values
x = raw_train.iloc[:, 1:].values.astype('float32') / 255

y = raw_train.iloc[:, 0] # labels
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=42)
x_train = x_train.reshape(-1, 28, 28,1)

x_val = x_val.reshape(-1, 28, 28,1)

y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),    

    

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),##

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256),

    tf.keras.layers.LeakyReLU(alpha=0.1),

 

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10, activation='softmax')

])
optimizer = RMSprop(learning_rate=0.002,

    rho=0.9,

    momentum=0.1,

    epsilon=1e-07,

    centered=True,

    name='RMSprop')

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])
batch_size = 1024

num_classes = 10

epochs = 40
history = model.fit(x_train, y_train, batch_size=batch_size,

                              epochs=15,

                              validation_data=(x_val, y_val))
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
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

raw_test = pd.read_csv('../input/Kannada-MNIST/test.csv')

raw_test_id=raw_test.id

raw_test=raw_test.drop("id",axis="columns")

raw_test=raw_test / 255

test=raw_test.values.reshape(-1,28,28,1)

test.shape
sub=model.predict(test)     ##making prediction

sub=np.argmax(sub,axis=1) ##changing the prediction intro labels



sample_sub['label']=sub

sample_sub.to_csv('submission.csv',index=False)