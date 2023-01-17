from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Activation

# Images fed into this model are 512 x 512 pixels with 3 channels
img_shape = (512,512,3)

# Set up model
model = Sequential()

# Add convolutional layer with 3, 3 by 3 filters and a stride size of 1
# Set padding so that input size equals output size
model.add(Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same',input_shape=img_shape))
# Add relu activation to the layer 
model.add(Activation('relu'))

# The same conv layer in a more common notation
model.add(Conv2D(3,(3,3),padding='same',activation='relu'))
# Give out model summary to show that both layers are the same
model.summary()
