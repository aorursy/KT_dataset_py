import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt



%matplotlib inline
!ls ../input/
types = ['containership', 'cruiser', 'destroyer','coastguard', 'smallfish', 'methanier', 'cv', 'corvette', 'submarine', 'tug']

types_id = dict(zip(types, range(len(types))))



ships = np.load('../input/ships.npz')

X_data = ships['X']

Y_data = ships['Y']
from keras.utils import np_utils



X_train = X_data

Y_train = np_utils.to_categorical(Y_data)
i = np.random.randint(len(Y_data))

print("Ship #%d is a %s" % (i,types[Y_data[i]]))

print(Y_train[i])

plt.imshow(X_data[i])
from keras.models import Model

from keras.layers import Input, Dense, Dropout, Flatten



inputs = Input(shape=X_train[0].shape, name='cnn_input')

x = Flatten()(inputs)

x = Dense(256, activation='relu')(x)

x = Dropout(0.5)(x)

outputs = Dense(len(types), activation='softmax')(x)



model = Model(inputs, outputs)
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy']

              )



model.fit(X_train, Y_train, epochs=1, batch_size=8, validation_split=0.1)
ships = np.load('../input/ships_test.npz')

X_test = ships['X']

Y_test = np_utils.to_categorical(ships['Y'])
# predict results

res = model.predict(X_test).argmax(axis=1)

df = pd.DataFrame({"Category":res})

df.to_csv("reco_nav.csv", index_label="Id")
