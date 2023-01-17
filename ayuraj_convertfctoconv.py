from keras.applications import VGG16

from keras import backend as K

from keras.models import Model

from keras.layers import Conv2D, Conv2DTranspose

from keras.layers import Input
K.clear_session()
model = VGG16(weights='imagenet')
model.summary()
model.get_layer('fc1')
model.layers.pop()
model.summary()
model.layers.pop()
model.summary()
model.layers.pop()

model.layers.pop()
model.summary()
model.layers[-1].output
fc2conv1 = Conv2D(4096, kernel_size=[7,7], strides=(1,1), padding='valid', activation='relu')(model.layers[-1].output)
fc2conv2 = Conv2D(4096, kernel_size=[1,1], strides=(1,1), padding='valid', activation='relu')(fc2conv1)
fc2conv3 = Conv2D(1000, kernel_size=[1,1], strides=(1,1), padding='valid', activation='relu')(fc2conv2)
fcnModel = Model(inputs=model.input, outputs=fc2conv3)
fcnModel.summary()