# importing libraries
import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# nn
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Input, Add
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import preprocess_input
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, Callback

import warnings
warnings.filterwarnings('ignore')
image_dir = "../input/font-recognition-data/Font Dataset Large/"
print("Total number of Images are: {}".format(len(glob(image_dir + "*/*"))))
class ResNet50V2():
    @staticmethod
    def build(height = 224, width = 224, depth = 3, classes = 1000):
        bn_axis = 3
        input_shape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            bn_axis = 1

        img_input = Input(shape = input_shape)
        x = ZeroPadding2D(padding=((3, 3), (3, 3)), name = 'conv1_pad')(img_input)
        x = Conv2D(64, 7, strides = 2, name = "conv1_conv")(x)
        x = ZeroPadding2D(padding = ((1, 1), (1, 1)), name = "pool1_pad")(x)
        x = MaxPooling2D(3, strides = 2, name = "pool1_pool")(x)

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block1_preact_bn")(x)
        preact = Activation("relu", name = "conv2_block1_preact_relu")(preact)

        shortcut = Conv2D(4 * 64, 1, strides = 1, name = "conv2_block1_0_conv")(preact)
        x = Conv2D(64, 1, strides = 1, use_bias = False, name = "conv2_block1_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block1_1_bn")(x)
        x = Activation("relu",  name = "conv2_block1_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv2_block1_2_pad")(x)
        x = Conv2D(64, 3, strides = 1, use_bias = False, name = "conv2_block1_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block1_2_bn")(x)
        x = Activation("relu", name = "conv2_block1_2_relu")(x)

        x = Conv2D(4 * 64, 1, name = "conv2_block1_3_conv")(x)
        x = Add(name = "conv2_block1_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block2_preact_bn")(x)
        preact = Activation("relu", name = "conv2_block2_preact_relu")(preact)

        shortcut = x
        x = Conv2D(64, 1, strides = 1, use_bias = False, name = "conv2_block2_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block2_1_bn")(x)
        x = Activation("relu",  name = "conv2_block2_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv2_block2_2_pad")(x)
        x = Conv2D(64, 3, strides = 1, use_bias = False, name = "conv2_block2_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block2_2_bn")(x)
        x = Activation("relu", name = "conv2_block2_2_relu")(x)

        x = Conv2D(4 * 64, 1, name = "conv2_block2_3_conv")(x)
        x = Add(name = "conv2_block2_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block3_preact_bn")(x)
        preact = Activation("relu", name = "conv2_block3_preact_relu")(preact)

        shortcut = MaxPooling2D(1, strides = 2)(x)
        x = Conv2D(64, 1, strides = 1, use_bias = False, name = "conv2_block3_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block3_1_bn")(x)
        x = Activation("relu",  name = "conv2_block3_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv2_block3_2_pad")(x)
        x = Conv2D(64, 3, strides = 2, use_bias = False, name = "conv2_block3_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv2_block3_2_bn")(x)
        x = Activation("relu", name = "conv2_block3_2_relu")(x)

        x = Conv2D(4 * 64, 1, name = "conv2_block3_3_conv")(x)
        x = Add(name = "conv2_block3_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block1_preact_bn")(x)
        preact = Activation("relu", name = "conv3_block1_preact_relu")(preact)

        shortcut = Conv2D(4 * 128, 1, strides = 1, name = "conv3_block1_0_conv")(preact)
        x = Conv2D(128, 1, strides = 1, use_bias = False, name = "conv3_block1_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block1_1_bn")(x)
        x = Activation("relu",  name = "conv3_block1_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv3_block1_2_pad")(x)
        x = Conv2D(128, 3, strides = 1, use_bias = False, name = "conv3_block1_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block1_2_bn")(x)
        x = Activation("relu", name = "conv3_block1_2_relu")(x)

        x = Conv2D(4 * 128, 1, name = "conv3_block1_3_conv")(x)
        x = Add(name = "conv3_block1_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block2_preact_bn")(x)
        preact = Activation("relu", name = "conv3_block2_preact_relu")(preact)

        shortcut = x
        x = Conv2D(128, 1, strides = 1, use_bias = False, name = "conv3_block2_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block2_1_bn")(x)
        x = Activation("relu",  name = "conv3_block2_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv3_block2_2_pad")(x)
        x = Conv2D(128, 3, strides = 1, use_bias = False, name = "conv3_block2_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block2_2_bn")(x)
        x = Activation("relu", name = "conv3_block2_2_relu")(x)

        x = Conv2D(4 * 128, 1, name = "conv3_block2_3_conv")(x)
        x = Add(name = "conv3_block2_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block3_preact_bn")(x)
        preact = Activation("relu", name = "conv3_block3_preact_relu")(preact)

        shortcut = x
        x = Conv2D(128, 1, strides = 1, use_bias = False, name = "conv3_block3_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block3_1_bn")(x)
        x = Activation("relu",  name = "conv3_block3_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv3_block3_2_pad")(x)
        x = Conv2D(128, 3, strides = 1, use_bias = False, name = "conv3_block3_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block3_2_bn")(x)
        x = Activation("relu", name = "conv3_block3_2_relu")(x)

        x = Conv2D(4 * 128, 1, name = "conv3_block_3_conv")(x)
        x = Add(name = "conv3_block3_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block4_preact_bn")(x)
        preact = Activation("relu", name = "conv3_block4_preact_relu")(preact)

        shortcut = MaxPooling2D(1, strides = 2)(x)
        x = Conv2D(128, 1, strides = 1, use_bias = False, name = "conv3_block4_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block4_1_bn")(x)
        x = Activation("relu",  name = "conv3_block4_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv3_block4_2_pad")(x)
        x = Conv2D(128, 3, strides = 2, use_bias = False, name = "conv3_block4_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv3_block4_2_bn")(x)
        x = Activation("relu", name = "conv3_block4_2_relu")(x)

        x = Conv2D(4 * 128, 1, name = "conv3_block4_3_conv")(x)
        x = Add(name = "conv3_block4_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block1_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block1_preact_relu")(preact)

        shortcut = Conv2D(4 * 256, 1, strides = 1, name = "conv4_block1_0_conv")(preact)
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block1_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block1_1_bn")(x)
        x = Activation("relu",  name = "conv4_block1_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block1_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block1_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block1_2_bn")(x)
        x = Activation("relu", name = "conv4_block1_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block1_3_conv")(x)
        x = Add(name = "conv4_block1_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block2_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block2_preact_relu")(preact)

        shortcut = x
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block2_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block2_1_bn")(x)
        x = Activation("relu",  name = "conv4_block2_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block2_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block2_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block2_2_bn")(x)
        x = Activation("relu", name = "conv4_block2_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block2_3_conv")(x)
        x = Add(name = "conv4_block2_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block3_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block3_preact_relu")(preact)

        shortcut = x
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block3_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block3_1_bn")(x)
        x = Activation("relu",  name = "conv4_block3_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block3_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block3_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block3_2_bn")(x)
        x = Activation("relu", name = "conv4_block3_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block3_3_conv")(x)
        x = Add(name = "conv4_block3_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block4_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block4_preact_relu")(preact)

        shortcut = x
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block4_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block4_1_bn")(x)
        x = Activation("relu",  name = "conv4_block4_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block4_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block4_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block4_2_bn")(x)
        x = Activation("relu", name = "conv4_block4_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block4_3_conv")(x)
        x = Add(name = "conv4_block4_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block5_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block5_preact_relu")(preact)

        shortcut = x
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block5_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block5_1_bn")(x)
        x = Activation("relu",  name = "conv4_block5_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block5_2_pad")(x)
        x = Conv2D(256, 3, strides = 1, use_bias = False, name = "conv4_block5_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block5_2_bn")(x)
        x = Activation("relu", name = "conv4_block5_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block5_3_conv")(x)
        x = Add(name = "conv4_block5_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block6_preact_bn")(x)
        preact = Activation("relu", name = "conv4_block6_preact_relu")(preact)

        shortcut = MaxPooling2D(1, strides = 2)(x)
        x = Conv2D(256, 1, strides = 1, use_bias = False, name = "conv4_block6_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block6_1_bn")(x)
        x = Activation("relu",  name = "conv4_block6_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv4_block6_2_pad")(x)
        x = Conv2D(256, 3, strides = 2, use_bias = False, name = "conv4_block6_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv4_block6_2_bn")(x)
        x = Activation("relu", name = "conv4_block6_2_relu")(x)

        x = Conv2D(4 * 256, 1, name = "conv4_block6_3_conv")(x)
        x = Add(name = "conv4_block6_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block1_preact_bn")(x)
        preact = Activation("relu", name = "conv5_block1_preact_relu")(preact)

        shortcut = Conv2D(4 * 512, 1, strides = 1, name = "conv5_block1_0_conv")(preact)
        x = Conv2D(512, 1, strides = 1, use_bias = False, name = "conv5_block1_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block1_1_bn")(x)
        x = Activation("relu",  name = "conv5_block1_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv5_block1_2_pad")(x)
        x = Conv2D(512, 3, strides = 1, use_bias = False, name = "conv5_block1_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block1_2_bn")(x)
        x = Activation("relu", name = "conv5_block1_2_relu")(x)

        x = Conv2D(4 * 512, 1, name = "conv5_block1_3_conv")(x)
        x = Add(name = "conv5_block1_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block2_preact_bn")(x)
        preact = Activation("relu", name = "conv5_block2_preact_relu")(preact)

        shortcut = x
        x = Conv2D(512, 1, strides = 1, use_bias = False, name = "conv5_block2_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block2_1_bn")(x)
        x = Activation("relu",  name = "conv5_block2_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv5_block2_2_pad")(x)
        x = Conv2D(512, 3, strides = 1, use_bias = False, name = "conv5_block2_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block2_2_bn")(x)
        x = Activation("relu", name = "conv5_block2_2_relu")(x)

        x = Conv2D(4 * 512, 1, name = "conv5_block2_3_conv")(x)
        x = Add(name = "conv5_block2_out")([shortcut, x])

        preact = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block3_preact_bn")(x)
        preact = Activation("relu", name = "conv5_block3_preact_relu")(preact)

        shortcut = x    
        x = Conv2D(512, 1, strides = 1, use_bias = False, name = "conv5_block3_1_conv")(preact)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block3_1_bn")(x)
        x = Activation("relu",  name = "conv5_block3_1_relu")(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name = "conv5_block3_2_pad")(x)
        x = Conv2D(512, 3, strides = 1, use_bias = False, name = "conv5_block3_2_conv")(x)
        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "conv5_block3_2_bn")(x)
        x = Activation("relu", name = "conv5_block3_2_relu")(x)

        x = Conv2D(4 * 512, 1, name = "conv5_block3_3_conv")(x)
        x = Add(name = "conv5_block3_out")([shortcut, x])


        x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = "post_bn")(x)
        x = Activation("relu", name = "post_relu")(x)

        x = GlobalAveragePooling2D(name = "avg_pool")(x)
        x = Dense(classes, activation = "softmax", name = "probs")(x)

        model = Model(img_input, x, name = "ResNet50V2")
        return model
model = ResNet50V2.build(224, 224, 3, 48)
print(model.summary())
# image augmentation
datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                             rotation_range = 20,
                             zoom_range = 0.15,
                             shear_range = 0.15,
                             horizontal_flip = True,
                             rescale = 1./255,
                             validation_split = 0.15)
batch_size = 32
data_dir = "../input/font-recognition-data/Font Dataset Large"

training_set = datagen.flow_from_directory(data_dir,
                                           target_size = (224, 224),
                                           batch_size = batch_size,
                                           subset = "training",
                                           class_mode = "categorical")
test_set = datagen.flow_from_directory(data_dir,
                                       target_size = (224, 224),
                                       batch_size = batch_size,
                                       subset = "validation",
                                       class_mode = 'categorical')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                              factor = 0.1,
                              patience = 2, 
                              min_lr=0.00001,
                              verbose = 1)
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True
    
callbacks = myCallback()
print(training_set.class_indices)

opt = Adam(lr = 0.001)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
H = model.fit(training_set,
              steps_per_epoch = training_set.samples//batch_size,
              validation_data = test_set,
              epochs = 10,
              validation_steps = test_set.samples//batch_size,
              callbacks = [reduce_lr, callbacks],
              verbose = 1)
print("[Info] serializing network...")
model.save("resnet50v2.h5")
import matplotlib
matplotlib.use("Agg")
%matplotlib inline

print("[Info] visualising model...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 5), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 5), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 5), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 5), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
plt.savefig("resnet50v2.png")
