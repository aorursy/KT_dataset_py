# imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
# load labels and read in image information

# the images are small enough to load them all into the memory rather than use an image-loading generator

path = '../input/train/train/'

df = pd.read_csv('../input/train.csv')

n = len(df)

y = np.zeros(shape=(n,1),dtype=int)

X = np.zeros(shape=(n,32,32,3))



for i,v in df.iterrows():

    y[i] = v[1]

    X[i] = (mpimg.imread(f'{path}/{v[0]}'))/255
# show one image

i = np.random.randint(len(X))

plt.imshow(X[i])

plt.show()

print(y[i])
# add permuations (flips and rotation)

X_vflip = np.zeros(shape=(n,32,32,3))

X_hflip = np.zeros(shape=(n,32,32,3))

X_rot = np.zeros(shape=(n,32,32,3))

for i in range(n):

    X_vflip[i] = np.flip(X[i],axis=0)

    X_hflip[i] = np.flip(X[i],axis=1)

    X_rot[i] = np.rot90(X[i])
# observe a random permutation:

fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(12,6))

i = np.random.randint(len(X))

ax[0,0].imshow(X[i])

ax[1,0].imshow(X_vflip[i])

ax[0,1].imshow(X_hflip[i])

ax[1,1].imshow(X_rot[i])

plt.show()

print(y[i])
# after initial testing, there is no more need for testing, and the entire dataset will probably yield better predictions



#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# combined permuations with data:

X_all = np.concatenate((X,X_vflip,X_hflip,X_rot))

y_all = np.concatenate((y,y,y,y))
# generate model

from keras.models import Model

from keras.layers import Input, Dense, Convolution2D, Flatten, MaxPooling2D, Dropout
input_layer = Input(shape=(32,32,3))

x = Convolution2D(16, kernel_size=3,activation='relu')(input_layer)

x = MaxPooling2D()(x)

x = Convolution2D(32, kernel_size=3,activation='relu')(x)

x = Convolution2D(64, kernel_size=3,activation='relu')(x)

x = MaxPooling2D()(x)

x = Dense(256,activation='relu')(Flatten()(x))

x = Dropout(rate=0.5)(x)

output = Dense(1, activation='sigmoid')(x)



model = Model(inputs=input_layer, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()
# train model on all data including permutations

model.fit(X_all,y_all,epochs=10)
# originally was used on the test set, but now that I have used the entire dataset, this is not useful

#model.evaluate(X_test,y_test)
# read in submission (test) images

tpath = '../input/test/test/'

submit = pd.read_csv('../input/sample_submission.csv')

t_X = np.zeros(shape=(len(submit),32,32,3))



for i,v in submit.iterrows():

    t_X[i] = (mpimg.imread(f'{tpath}/{v[0]}'))/255
# get predictions

t_y = model.predict(t_X)
# insert predictions in submission file

submit['has_cactus'] = t_y

submit.head()
# save output for submission

submit.to_csv('170419_perm.csv',index=False)