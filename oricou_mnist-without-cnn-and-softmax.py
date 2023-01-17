import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import tensorflow as tf



np.set_printoptions(precision=4, suppress=True, floatmode='fixed')



SEED = 123                 # to be able to rerun the same NN

np.random.seed(SEED)

tf.set_random_seed(SEED)



%matplotlib inline
data = np.load("../input/mnist.npz")



x_train = data['x_train']

y_train = data['y_train']

x_test = data['x_test']

y_test = data['y_test']



x_train, x_test = x_train / 255.0, x_test / 255.0
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten



model = Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy']

              )



model.fit(x_train, y_train, epochs=10, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])
res = model.predict(x_test)

res = pd.DataFrame({'true':y_test, 'guess':np.argmax(res, axis=1), 'trust':np.max(res, axis=1)})
bad = res[y_test != res.guess].sort_values('trust', ascending=False)

bad.head(10)
i = bad.index.values[0]

res = model.predict(x_test[i][None,:,:])  # None allows to add a dimension, Error messages told me to do that :-)

print("Image", i)

print(f"Model says it is a {np.argmax(res)} while it is a {y_test[i]}")

print("Stats are", np.array(res))

plt.imshow(x_test[i])