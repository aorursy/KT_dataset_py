# Set up code checking

#from learntools.core import binder

#binder.bind(globals())

#from learntools.deep_learning.exercise_4 import *

#print("Setup Complete")
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D



num_classes = 2

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



my_new_model.layers[0].trainable = False



#step_1.check()

#step_1.hint()

# step_1.solution()
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#step_2.solution()
#step_3.solution()
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224 

data_generator = ImageDataGenerator(preprocess_input)

train_generator = data_generator.flow_from_directory(directory = '../input/dogs-gone-sideways/images/train', target_size=(image_size, image_size), batch_size=10, class_mode='categorical')

test_generator = data_generator.flow_from_directory(directory = '../input/dogs-gone-sideways/images/val', target_size=(image_size, image_size), class_mode='categorical')



fit_stats = my_new_model.fit(train_generator, epochs=30, steps_per_epoch= 10, validation_data=test_generator, validation_steps=5)

import matplotlib.pyplot as plt

def plot_conv_weights(modell, layer_index):

    W = modell.get_layer(index=layer_index).get_weights()[0]

    if len(W.shape) == 4:

        #W = np.squeeze(W)

        W = W.reshape((W.shape[0], W.shape[1], W.shape[2]*W.shape[3]))

        fig, axs = plt.subplots(5,5, figsize=(8,8))

        fig.subplots_adjust(hspace = .5, wspace=.001)

        axs = axs.ravel()

        for i in range(25):

            axs[i].imshow(W[:,:,i], cmap='gray')

            axs[i].set_title(str(i))

plot_conv_weights(my_new_model, 0)

plot_conv_weights(my_new_model, 1)
step_4.solution()