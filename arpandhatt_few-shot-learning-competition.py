import numpy as np
import pandas as pd
from skimage.io import imshow
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
npz = np.load('../input/input_data.npz')
X_train = npz['X_train'][:1000]
Y_train = npz['Y_train'][:1000]
del npz
ix = 100 #0-3999
imshow(np.squeeze(X_train[ix,:,:,2]))#Looking at the combined channel
plt.show()
labels = ['Ship','Iceberg']#0 is no iceberg(ship) and 1 is iceberg
print ('This is:',labels[int(Y_train[ix])])
model = Sequential([
    #layerrrsssss
])
learning_rate = 0.01#change it if you'd like
optimizer = Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
model.summary()
#We will train on 900 examples and validate on 100
model.fit(X_train[:900], Y_train[:900],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train[900:], Y_train[900:]))
p = model.predict(X_train[900:], verbose=1)
ix = 6
imshow(np.squeeze(X_train[900+ix,:,:,2]))
plt.show()
print ('Probability:')
for i in range(1):
    print ('|'+'\u2588'*int(p[ix,i]*50)+' '*int((1-p[ix,i])*50)+'| Iceberg'+' {:.5f}%'.format(p[ix,i]*100))
print ('This is:',labels[int(Y_train[900+ix])])
