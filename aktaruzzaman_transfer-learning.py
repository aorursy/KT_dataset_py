import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
from keras.layers import Input, Dense,Conv2D

input_ = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=input_)
model.summary()

number_of_class=3
vgg16_layer = model.get_layer('block1_conv2').output
out_layer = Conv2D(3, (1, 1), activation='relu')(vgg16_layer)
#out_layer = Dense(number_of_class, activation='softmax',name = 'custome_layer1')(vgg16_layer)

My_model = Model(input_,out_layer)
My_model.summary()
#image load by using keras model
from keras.preprocessing import image
def im_read(impaths):
    df = []
    for path_ in impaths:
        img = image.load_img(path=path_,target_size=(224,224,3))
        img = image.img_to_array(img)
        df.append(img)
    return np.array(df, dtype=float)

#image load by using keras model
from keras.preprocessing import image
def im_readX(impaths):
    df = []
    for path_ in impaths:
        img = image.load_img(path=path_,target_size=(224,224,3))
        img = image.img_to_array(img)
        df.append(img)
    return np.array(df, dtype=float)
path = pd.Series(['../input/cat.jpg'])
Y = im_read(path)
X = im_readX(path)
X.shape, Y.shape
X[...,0][0]
plt.imshow(X[...,0][0])
for layer in My_model.layers[:-1]:
    layer.trainable = False
My_model.summary()

X.shape
from keras import optimizers
opt = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
My_model.compile(optimizer = opt , loss = 'mse', metrics=['mae','accuracy'])
My_model.fit(X, Y, epochs=10, batch_size=30)
plt.imshow(My_model.predict(X)[...,0][0])


