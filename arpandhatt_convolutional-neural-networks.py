import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.io import imshow
#Read CSV
csv = pd.read_csv('../input/fashion-mnist_train.csv')
#Separate into matricies
X_train = csv.iloc[:,1:786].as_matrix()
Y_train = csv.iloc[:,0].as_matrix()
# This is very simple
X_train_imgs = np.zeros([60000,28,28,1])
for i in range(X_train.shape[0]):
    img = X_train[i,:].reshape([28,28,1])/255.
    X_train_imgs[i] = img
#oh stands for one-hot
#There are 60000 examples and 10 different pieces of clothing
Y_train_oh = np.zeros([60000,10])
for i in range(Y_train.shape[0]):
    oh = np.zeros([10])
    oh[int(Y_train[i])] = 1.
    Y_train_oh[i] = oh
ix = 59999 #0-59999
imshow(np.squeeze(X_train_imgs[ix]))
plt.show()
clothing = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print ('This is:',clothing[int(Y_train[ix])])
# Make sure you separate layers with commas!
model = Sequential([
    #layers
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#We will train on 59000 examples and validate on 1000
model.fit(X_train_imgs[:59000], Y_train_oh[:59000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[59001:], Y_train_oh[59001:]))
#First, let's get all the predictions
p = model.predict(X_train_imgs[59000:],verbose=1)
ix = 50
imshow(np.squeeze(X_train_imgs[59000+ix]))
plt.show()
print ('Probabilities:')
for i in range(10):
    print ('|'+'\u2588'*int(p[ix,i]*50)+clothing[i]+' {:.5f}%'.format(p[ix,i]*100))
