import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


# imports the MobileNet model and discards the 1000-category output layer
# the output layer is not useful to us unless we happen to have the exact same 1000 classes!
base_model=MobileNet(weights='imagenet', include_top=False)


# define our own fully connected layers
# these will stack after the convolutional layers, which are borrowed from MobileNet
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
out=Dense(120,activation='softmax')(x)


# now define our model by chaining our custom output layers on the MobileNet convolutional core
model=Model(inputs=base_model.input, outputs=out)


# optionally set the first 20 layers of the network (the MobileNet component) to be non-trainable
# this means that we will use (the convolutional part of) MobileNet exactly
# which will speed up training. The alternative would be to continue to optimize these layers
for layer in model.layers[:20]:
    layer.trainable=False
    

# finally, compile the model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# now we are ready for fitting!