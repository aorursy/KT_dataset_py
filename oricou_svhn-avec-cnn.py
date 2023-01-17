import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import tensorflow as tf



SEED = 123                 # to be able to rerun the same NN

np.random.seed(SEED)

tf.set_random_seed(SEED)



np.set_printoptions(precision=4, suppress=True, floatmode='fixed')



%matplotlib inline
!nvidia-smi
!lscpu
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



!ls -lh ../input/



# Any results you write to the current directory are saved as output.
import scipy.io as sio



train_data = sio.loadmat('../input/train_32x32.mat')

test_data = sio.loadmat('../input//test_32x32.mat')

extra_data = sio.loadmat('../input/extra_32x32.mat')



X_train, y_train = train_data['X'], train_data['y']

X_test, y_test = test_data['X'], test_data['y']

X_extra, y_extra = extra_data['X'], extra_data['y']



classes = [0,1,2,3,4,5,6,7,8,9]

nb_classes = 10



print(X_train.shape, X_test.shape, X_extra.shape)
# on réordonne pour correspondre à l'ordre de Tensorflow

X_train = np.transpose(X_train,(3,0,1,2))

X_test = np.transpose(X_test,(3,0,1,2))

X_extra = np.transpose(X_extra,(3,0,1,2))



# on fusionne les données de base avec les extras

X_train = np.concatenate([X_train, X_extra])

y_train = np.concatenate([y_train, y_extra])



# et on normalise

X_train = X_train.astype('float32') / 255

X_test = X_test.astype('float32') / 255
from keras.utils import to_categorical



print(y_train[:4])

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

y_train[:4]
i = np.random.randint(1, len(X_train))

print("Label %d is" % i, y_train[i])

plt.imshow(X_train[i])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



model = Sequential()



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(len(y_train[0]), activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adadelta',      

              metrics=['accuracy'])
model.summary()
model_history = model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split = 0.1)
score = model.evaluate(X_test, y_test, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])
res_test = model.predict(X_test)
res_test = pd.DataFrame({'true':np.argmax(y_test, axis=1), 'guess':np.argmax(res_test, axis=1), 'trust':np.max(res_test, axis=1)})

res_test.head(10)
errors = res_test[res_test.true != res_test.guess].sort_values('trust', ascending=False)

errors.head(10)
print('Percentage of error %4.2f %%' % (100 * len(errors)/len(X_test))) # on vérifie que c'est bien le résultat donné ci-dessus
i = 15700 #1318

res = model.predict(X_test[i][None,:,:])  # None permet d'augmenter la dimension du tableau (sans on a un message d'erreur clair)

print("Image", i)

print(f"Model says it is a {np.argmax(res)} while it is a {np.argmax(y_test[i])}")

print("Stats are", np.array(res))

plt.imshow(X_test[i])