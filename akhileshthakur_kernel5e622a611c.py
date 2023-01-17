import keras

from keras.layers import Dense, Dropout , Flatten

from keras.layers import Conv2D, MaxPooling2D, Activation

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers



from skimage import io, transform



import os, glob

import cv2

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


train_images = glob.glob("../input/fingers/train/*.png")

test_images = glob.glob("../input/fingers/test/*.png")
im = cv2.imread(train_images[0])

#rgb = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)

print(im.shape)

io.imshow(im)

plt.show()
X_train = []

X_test = []

y_train = []

y_test = []



for img in train_images :

    img_read = cv2.imread(img)

    #img_rgb = cv2.cvtColor(img_read,cv2. COLOR_GRAY2RGB)

    X_train.append(img_read)

    y_train.append(img[-6:-5])

    

for img in test_images:

    img_read = cv2.imread(img)

    X_test.append(img_read)

    y_test.append(img[-6:-5])



print(y_train[:5])

io.imshow(X_train[0])

plt.show()

io.imshow(X_train[1])

plt.show()

io.imshow(X_train[2])

plt.show()
X_train = np.array(X_train)

X_test = np.array(X_test)

print(X_train.shape , X_test.shape)
print(X_train.shape , X_test.shape)
label_to_int = {

    '0':0,

    '1':1,

    '2':2,

    '3':3,

    '4':4,

    '5':5

}
temp = []

for label in y_train:

    temp.append(label_to_int[label])

y_train = temp.copy()



temp = []

for label in y_test:

    temp.append(label_to_int[label])

y_test = temp.copy()
y_train = keras.utils.to_categorical(y_train, num_classes = 6)

y_test = keras.utils.to_categorical(y_test, num_classes = 6)
model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False,input_shape=(128,128,3))
model.summary()
x = model.output
x = MaxPooling2D()(x)

x = Dense(1024,activation='relu')(x)

x = Dense(1024,activation='relu')(x)

x = Dense(512,activation='relu')(x)

predictions = Dense(6,activation='softmax')(x)
nw_model = keras.Model(inputs = model.input , outputs=predictions)

nw_model.summary()
for layer in nw_model.layers[:20]:

    layer.trainable=False

for layer in nw_model.layers[20:]:

    layer.trainable=True
nw_model.compile(optimizer=keras.optimizers.Adam(.0001),loss='categorical_crossentropy',metrics=['accuracy'])
nw_model.fit(x=X_train , y=y_train , batch_size=64 , validation_data = (X_test,y_test), epochs = 10)