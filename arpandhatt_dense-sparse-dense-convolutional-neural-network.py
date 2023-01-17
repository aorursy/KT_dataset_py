import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.io import imshow
from keras import backend as K
from keras.constraints import Constraint
#Read CSV
csv = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
#Separate into matricies
X_train = csv.iloc[:,1:786].as_matrix()
Y_train = csv.iloc[:,0].as_matrix()
# This is very simple
X_train_imgs = np.zeros([X_train.shape[0],28,28,1])
for i in range(X_train.shape[0]):
    img = X_train[i,:].reshape([28,28,1])/255.
    X_train_imgs[i] = img
#oh stands for one-hot
#There are 60000 examples and 10 different pieces of clothing
Y_train_oh = np.zeros([Y_train.shape[0],10])
for i in range(Y_train.shape[0]):
    oh = np.zeros([10])
    oh[int(Y_train[i])] = 1.
    Y_train_oh[i] = oh
ix = 12345 #0-41999
imshow(np.squeeze(X_train_imgs[ix]))
plt.show()
label = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print ('This is:',label[int(Y_train[ix])])
#Let's code our own constraint!
class Sparse(Constraint):
    '''
    We will use one variable: Mask
    After we train our model dense model,
    we will save the weights and analyze them.
    We will create a mask where 1 means the
    number is far away enough from 0 and 0
    if it is to close to 0. We will multiply
    the weights by 0(making them 0) if they
    are supposed to be masked.
    '''
    
    def __init__(self, mask):
        self.mask = K.cast_to_floatx(mask)
    
    def __call__(self,x):
        return self.mask * x
    
    def get_config(self):
        return {'mask': self.mask}
# Make sure you separate layers with commas!
model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1)),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250),
    Activation('relu'),
    Dropout(0.4),
    Dense(10),
    Activation('softmax')
])
adam = Adam()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()
#We will train on 41000 examples and validate on 18999(To be quick)
model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))
def create_sparsity_masks(model,sparsity):
    weights_list = model.get_weights()
    masks = []
    for weights in weights_list:
        #We can ignore biases
        if len(weights.shape) > 1:
            weights_abs = np.abs(weights)
            masks.append((weights_abs>np.percentile(weights_abs,sparsity))*1.)
    return masks
masks = create_sparsity_masks(model,30)#Closest 30% to 0
sparse_model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1), kernel_constraint=Sparse(masks[0])),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3, kernel_constraint=Sparse(masks[1])),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250, kernel_constraint=Sparse(masks[2])),
    Activation('relu'),
    Dropout(0.4),
    Dense(10, kernel_constraint=Sparse(masks[3])),
    Activation('softmax')
])

adam = Adam()
sparse_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
sparse_model.summary()
#Get weights from densely trained model
sparse_model.set_weights(model.get_weights())
sparse_model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))
redense_model = Sequential([
    Conv2D(32,3,input_shape=(28,28,1)),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,3),
    Activation('relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(250),
    Activation('relu'),
    Dropout(0.4),
    Dense(10),
    Activation('softmax')
])

adam = Adam(lr=0.0001)#Default Adam lr is 0.001 so I set it to 0.0001
redense_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
redense_model.summary()
#Get weights from sparsely trained model
redense_model.set_weights(sparse_model.get_weights())

redense_model.fit(X_train_imgs[:41000], Y_train_oh[:41000],
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(X_train_imgs[41001:], Y_train_oh[41001:]))
#First, let's get all the predictions
p = redense_model.predict(X_train_imgs[41000:],verbose=1)
ix = 300
imshow(np.squeeze(X_train_imgs[41000+ix]))
plt.show()
print ('Probabilities:')
i = 0
for i in range(10):
    correct = (Y_train[41000+ix] == i)*1
    print ('|'+'\u2588'*int(p[ix,i]*50)+' '+label[i]+' {:.5f}%'.format(p[ix,i]*100)+' <=='*correct)