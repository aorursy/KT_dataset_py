import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))





import cv2

import keras

from keras import backend as K

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, Input, BatchNormalization, ZeroPadding2D, Dropout

from keras.models import Sequential, Model

from tqdm import tqdm
base = '../input/cell_images/cell_images/'

para = os.listdir(base+'Parasitized')

nor = os.listdir(base+'Uninfected')
len(para), len(nor)
def image_reader(path):

    '''

    Image_reader:

        Takes image as input and returns the RGB image of the same read image

        The image is also resized into smaller shape and Dimension of 110x110.

    '''

    t=cv2.imread(path)

    t=cv2.cvtColor(t, cv2.COLOR_BGR2RGB)

    t=cv2.resize(t, (110,110))

    return t
X = []

Y = [] 



for x in tqdm(para):

    # Parasite variable 

    try:

        t = image_reader(base+'Parasitized/'+x)

        X.append(t)

        Y.append(1)

    except:

        pass

    

for x in tqdm(nor):

    # Non Parasite images variable

    try:

        t = image_reader(base+'Uninfected/'+x)

        X.append(t)

        Y.append(0)

    except:

        pass
X=np.array(X)

Y=np.array(Y)

Y_oh = keras.utils.to_categorical(Y, num_classes=2)

print(X.shape, Y_oh.shape)
# Show training images

np.random.seed(10)



concat_img = None

for i in range(10):

    idx = np.random.randint(X.shape[0])

    if concat_img is None:

        concat_img = X[idx]

    else:

        concat_img = np.concatenate([concat_img, X[idx]], axis=1)

plt.figure(figsize=(15, 5)) 

plt.imshow(concat_img)
train_x, test_x, train_y, test_y = train_test_split(X,Y_oh,test_size=0.1, shuffle=True)
X_input = Input((110,110,3))



# Zero-Padding: pads the border of X_input with zeroes

X = ZeroPadding2D((3, 3))(X_input)



# CONV -> BN -> RELU Block applied to X

X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)

X = BatchNormalization(axis = 3, name = 'bn0')(X)

X = Activation('relu')(X)



# MORE CONVS

X = MaxPooling2D((2, 2))(X)

#shortcut = X

X = Conv2D(32, (3, 3), strides = (1, 1), padding="same")(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)

X = Conv2D(32, (3, 3), strides = (1, 1), padding="same")(X)

X = BatchNormalization()(X)

#X = layers.add([X, shortcut])

X = Activation('relu')(X)



# MAXPOOL

X = MaxPooling2D((2, 2), name='max_pool')(X)



# FLATTEN X (means convert it to a vector) + FULLYCONNECTED

X = Flatten()(X)



# MORE DENSE

X = Dense(128)(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)

X = Dropout(0.5)(X)



X = Dense(2, activation='softmax', name='fc')(X)



# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

model = Model(inputs = X_input, outputs = X, name='HappyModel')
def get_f1(y_true, y_pred): #taken from old keras source code

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val
model.compile('SGD', 'categorical_crossentropy', ['acc', get_f1])
callback = keras.callbacks.ModelCheckpoint('best_malaria.h5', 

                                           monitor='val_get_f1', 

                                           verbose=1, 

                                           save_weights_only=True, 

                                           save_best_only=True, 

                                           mode='max')
history = model.fit(train_x, train_y, validation_split=0.1, epochs=10, batch_size=128, callbacks=[callback])
model.load_weights('best_malaria.h5')
np.bincount(test_y.argmax(1))
model.evaluate(test_x, test_y)
!wget https://ik.imagekit.io/1tspxkg7vx/3-1_3YucqGN2z8.jpg https://ik.imagekit.io/1tspxkg7vx/1-0__nHp6ya3h.jpg https://ik.imagekit.io/1tspxkg7vx/1-1_YyeldWIMn1.jpg https://ik.imagekit.io/1tspxkg7vx/2-0_NxWFVOR3m2.jpg https://ik.imagekit.io/1tspxkg7vx/2-1_jeECkgM4cG.jpg
evaluation=[]

evaluation.append(image_reader('./1-0__nHp6ya3h.jpg'))

evaluation.append(image_reader('./1-1_YyeldWIMn1.jpg'))

evaluation.append(image_reader('./2-0_NxWFVOR3m2.jpg'))

evaluation.append(image_reader('./2-1_jeECkgM4cG.jpg'))

evaluation.append(image_reader('./3-1_3YucqGN2z8.jpg'))

evaluation=np.asarray(evaluation); evaluation.shape
idx=4

t = test_x[idx].reshape(1,110, 110, 3)

# print(np.argmax(test_y[idx]), np.argmax(model.predict(t)))

plt.imshow(t[0])
res=model.predict(evaluation)

res.argmax(1)
model.save_weights('happy_weights.h5')