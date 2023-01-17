'''
A. Arrange your data first
==========================
    Download data from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia .

'''
#%%                                A. Call libraries

#        $ source activate theano
#        $ ipython
# OR in Windows
#       > conda activate tensorflow_env
#       > atom
# 0. Release memory
%reset -f

# 1.0 Data manipulation library
#     Install in 'tf' environment
#     conda install -c anaconda pandas
import pandas as pd
# 1.1 Call libraries for image processing
#     Another preprocessing option is text and sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

# 1.2, Libraries for building sequential CNN model
#      A model is composed of sequence of layered objects
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

# 1.3.Keras has three backend implementations available: the TensorFlow,
#    the Theano, and CNTK backend.
"""
What is a "backend"?
(http://faroit.com/keras-docs/1.2.0/backend/)
    Keras is a model-level library, providing high-level building blocks
    for developing deep learning models. It does not handle itself low-level
    operations such as tensor products, convolutions and so on. Instead,
    it relies on a specialized, well-optimized tensor manipulation library
    to do so, serving as the "backend engine" of Keras.
    List of low-level functions:
    https://www.tensorflow.org/api_docs/python/tf/keras/backend
"""
from tensorflow.keras import backend as K

# 1.4 Save CNN model configuration
from tensorflow.keras.models import model_from_json

# 1.5 OS related
import os

# 1.6 For ROC plotting
import matplotlib.pyplot as plt

# 1.7
import numpy as np
# conda install scikit-learn
from sklearn import metrics
import time
#from skimage import exposure           # Not used

# 1.8
# conda install -c anaconda pillow
#  Then deactivate and activate environment
#   This step is a must here
from PIL import Image                  # Needed in Windows

import os
print(os.listdir("../input"))
# 2.0 Data folder containing all imaages
main_dir =os.listdir("../input")
print(main_dir)
train_folder= "../input/chest-xray-pneumonia/chest_xray/train/"
val_folder = "../input/chest-xray-pneumonia/chest_xray/val/"
test_folder = "../input/chest-xray-pneumonia/chest_xray/test/"
train_folder= "../input/chest-xray-pneumonia/chest_xray/train/"
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
print(len(os.listdir(train_n))) #  1341
print(len(os.listdir(train_n))) #  1341
rand_norm=np.random.randint(0,len(os.listdir(train_n)))
rand_norm  #1313
norm_pic=os.listdir(train_n)[rand_norm]
norm_pic  #'NORMAL2-IM-1343-0001.jpeg'
print("normal picture title: " ,norm_pic)

norm_pic_address=train_n+norm_pic
norm_pic_address
# 2.4 For PNEUMONIA picture
rand_p=np.random.randint(0,len(os.listdir(train_p)))
sic_pic=os.listdir(train_p)[rand_norm]
print("normal picture title: " ,sic_pic)

sic_address = train_p+sic_pic
sic_address
# 2.5  load Images

norm_load=Image.open(norm_pic_address)
norm_load
sic_load=Image.open(sic_address)
sic_load

#2.6 plot the Images

f=plt.figure(figsize=(10,6))
a1=f.add_subplot(1,2,1)
img_plot=plt.imshow(norm_load)
a1.set_title('Normal')

a2=f.add_subplot(1,2,2)
img_plot=plt.imshow(sic_load)
a2.set_title('Pneumonia')
# 3. Create convnet model
#    con->relu->pool->con->relu->pool->flatten->fc->fc

# 4.1   Call model constructor and then pass on a list of layers
#        https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

cnn = Sequential()
cnn.add(Conv2D(
	             filters=32,                       # For every filter there is set of weights
	                                               # For each filter, one bias. So total bias = 32
	             kernel_size=(3, 3),               # For each filter there are 3*3=9 kernel_weights
	             strides = (1,1),                  # So output shape will be 148 X 148 (W-F+1).
	                                               # Default strides is 1 only
	             input_shape=(64,64,3),          # (150,150,3)
	             use_bias=True,                     # Default value is True
	             padding='same' ,                  # 'va;id' => No padding. This is default.
	             name="Ist_conv_layer"

	             )
         )
cnn.add(Activation('relu'))

cnn.add(MaxPool2D(pool_size = (2, 2)))
cnn.add(Conv2D(
	             filters=16,                       # For every filter there is set of weights
	                                               # For each filter, one bias. So total bias = 32
	             kernel_size=(3, 3),               # For each filter there are 3*3=9 kernel_weights
	             strides = (1,1),                  # So output shape will be 148 X 148 (W-F+1).
	                                               # Default strides is 1 only
	             use_bias=True,                     # Default value is True
	             padding='same',                   # 'va;id' => No padding. This is default.
	             name="II_conv_layer"

	             )
         )
cnn.add(Activation('relu'))
cnn.add(MaxPool2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(16))
cnn.add(Activation('relu'))
cnn.add(Dense(8))
cnn.add(Activation('relu'))
cnn.add(Dense(1))
cnn.add(Activation('sigmoid'))
cnn.compile(
              loss='binary_crossentropy',  # Metrics to be adopted by convergence-routine
              optimizer='adam',         # Strategy for convergence?
              metrics=['accuracy'])        # Metrics, I am interested in


train_datagen = ImageDataGenerator(
                              rescale=1. / 255,      # Normalize colour intensities in 0-1 range
                              shear_range=0.2,       # Shear varies from 0-0.2
                              zoom_range=0.2,
                              horizontal_flip=True,
                             # preprocessing_function=preprocess
                              )

train_set = train_datagen.flow_from_directory(
                                               train_folder,       # Data folder of train xray
                                               target_size=(64, 64),  # Resize images
                                               batch_size=16,  # Return images in batches
                                               class_mode='binary'   # Output labels will be 1D binary labels
                                                                     # [1,0,0,1]
                                                                     # If 'categorical' output labels will be
                                                                     # 2D OneHotEncoded: [[1,0],[0,1],[0,1],[1,0]]
                                                                     # If 'binary' use 'sigmoid' at output
                                                                     # If 'categorical' use softmax at output

                                                )
val_dtgen = ImageDataGenerator(rescale=1. / 255)

# 5.4.2 validation data

validation_generator = val_dtgen.flow_from_directory(
                                                     val_folder,
                                                     target_size=(64, 64),   # Resize images
                                                     batch_size=16,    # batch size to augment at a time
                                                     class_mode='binary'  # Return 1D array of class labels
                                                     )
test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

test_set = test_datagen.flow_from_directory(
                                               test_folder,       # Data folder of train xray
                                               target_size=(64, 64),  # Resize images
                                               batch_size=16,  # Return images in batches
                                               class_mode='binary'   # Output labels will be 1D binary labels
                                                                     # [1,0,0,1]
                                                                     # If 'categorical' output labels will be
                                                                     # 2D OneHotEncoded: [[1,0],[0,1],[0,1],[1,0]]
                                                                     # If 'binary' use 'sigmoid' at output
                                                                     # If 'categorical' use softmax at output

                                                )
                                                #Found 624 images belonging to 2 classes.
cnn.summary()
batch_size = 10
epochs = 4
start = time.time()   # 6 minutes
for e in range(epochs):
    #print('Epoch', e)
    batches = 0
    for x_batch, y_batch in train_set:
        cnn.fit(x_batch, y_batch)
        batches += 1
        print ("Epoch: {0} , Batches: {1}".format(e,batches))
        if batches > 100:    # 200 * 10 = 2000 images
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

end = time.time()
(end - start)/60
result = cnn.evaluate(validation_generator,
                                  verbose = 1,
                                  steps = 1        # How many batches
                                  )

result  