from IPython.display import YouTubeVideo,HTML
YouTubeVideo("IAQp2Zuqevc", width=800, height=500)
# importing libraries
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
%matplotlib inline

# nn
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Input, Concatenate, Dropout
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, Callback

#bokeh
from bokeh.models import ColumnDataSource, HoverTool, Panel, FactorRange
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.palettes import Spectral6

import warnings
warnings.filterwarnings('ignore')
# setup file structure
base_dir = "../input/chest-xray-pneumonia/chest_xray/"
train_dir = os.path.join(base_dir, "train/")
test_dir = os.path.join(base_dir, "test/")
val_dir = os.path.join(base_dir, "val/")
print("Number of images in Train is {}".format(len(glob(train_dir + "*/*"))))
print("Number of images in Test is {}".format(len(glob(test_dir + "*/*"))))
print("Number of images in Validation is {}".format(len(glob(val_dir + "*/*"))))
# Distribution of images with different category
Categories = ["Train", "Test", "Validation"]
Subcategories = ['Normal', 'Pneumonia']

Train = [1341, 390]
Test = [3875, 8]
Validation = [234, 8]

data = {'Categories':Categories,
        'Train':Train,
        'Test':Test,
        'Validation':Validation}

x = [(categories, subcategories) for categories in Categories for subcategories in Subcategories]
counts = sum(zip(data['Train'], data['Test'], data['Validation']), ())

source = ColumnDataSource(data=dict(x=x, counts=counts, color=Spectral6))

p = figure(x_range=FactorRange(*x), plot_height=400, plot_width=800, title="Distribution of images with different category",
           tools="hover, pan, box_zoom, wheel_zoom, reset, save", tooltips= ("@x: @counts"))

p.vbar(x='x', top='counts', width=0.9, color='color', legend_field="x", source=source)

p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_center"
output_notebook()
show(p)
Normal = glob(train_dir + "NORMAL/*")
Pneumonia = glob(test_dir + "PNEUMONIA/*")
# Extract 9 random images from normal
random_images = [Normal[i] for i in range(9)]

print('Display Normal Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(random_images[i])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()  
# Extract 9 random images from Pneumonia
random_images = [Pneumonia[i] for i in range(9)]

print('Display Pneumonia Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(random_images[i])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()  
images_shape = []

for k, image_path in enumerate(glob(train_dir + "*/*")):
    image = Image.open(image_path)
    images_shape.append(image.size)

images_shape_df = pd.DataFrame(data = images_shape, columns = ['H', 'W'], dtype='object')
images_shape_df['Size'] = '[' + images_shape_df['H'].astype(str) + ',' + images_shape_df['W'].astype(str) + ']'
images_shape_df.head()
print("We have {} types of different shapes in training images".format(len(list(images_shape_df['Size'].unique()))))
Normal = glob(train_dir + "NORMAL/*") + glob(test_dir + "NORMAL/*") + glob(val_dir + "NORMAL/*")
Pneumonia = glob(train_dir + "PNEUMONIA/*") + glob(test_dir + "PNEUMONIA/*") + glob(val_dir + "PNEUMONIA/*")
print("We have total {} Normal Images".format(len(Normal)))
print("We have total {} Pneumonia Images".format(len(Pneumonia)))
# For all normal images
Train_normal = Normal[:1266]
Test_normal = Normal[1266:1424]
Validation_normal = Normal[1424:]

# For all pneumonia images
Train_pneumonia = Pneumonia[:3418]
Test_pneumonia = Pneumonia[3418:3845]
Validation_pneumonia = Pneumonia[3845:]
# Make new directory to store our balanced data
"""
Train
    Normal
    Pneumonia
Test
    Normal
    Pneumonia
Validation
    Normal
    Pneumonia
"""

!mkdir Train
!mkdir Test
!mkdir Validation
!mkdir Train/Normal
!mkdir Train/Pneumonia
!mkdir Test/Normal
!mkdir Test/Pneumonia
!mkdir Validation/Normal
!mkdir Validation/Pneumonia
# copy all images from Train_pneumonia to Train/Pneumonia
for i in Train_pneumonia:
    shutil.copy(i, "Train/Pneumonia/")
print("Copied all images from Train_pneumonia to Train/Pneumonia")
    
# copy all images from Test_pneumonia to Test/Pneumonia
for i in Test_pneumonia:
    shutil.copy(i, "Test/Pneumonia/")
print("Copied all images from Test_pneumonia to Test/Pneumonia")
    
# copy all images from Validation_pneumonia to Validation/Pneumonia
for i in Validation_pneumonia:
    shutil.copy(i, "Validation/Pneumonia/")
print("Copied all images from Validation_pneumonia to Validation/Pneumonia")

# copy all images from Train_normal to Train/Normal
for i in Train_normal:
    shutil.copy(i, "Train/Normal/")
print("Copied all images from Train_normal to Train/Normal")

# copy all images from Test_normal to Test/Normal
for i in Test_normal:
    shutil.copy(i, "Test/Normal/")
print("Copied all images from Test_normal to Test/Normal")
    
# copy all images from Validation_normal to Validation/Normal
for i in Validation_normal:
    shutil.copy(i, "Validation/Normal/")
print("Copied all images from Validation_normal to Validation/Normal")
# setup file structure
train_dir = "Train/"
test_dir = "Test/"
val_dir = "Validation/"
# check new data distribution
print("Number of images in Train is {}".format(len(glob(train_dir + "*/*"))))
print("Number of images in Test is {}".format(len(glob(test_dir + "*/*"))))
print("Number of images in Validation is {}".format(len(glob(val_dir + "*/*"))))
# Distribution of images with different category
Categories = ["Train", "Test", "Validation"]
Subcategories = ['Normal', 'Pneumonia']

Train = [1266, 427]
Test = [3418, 159]
Validation = [158, 428]

data = {'Categories':Categories,
        'Train':Train,
        'Test':Test,
        'Validation':Validation}

x = [(categories, subcategories) for categories in Categories for subcategories in Subcategories]
counts = sum(zip(data['Train'], data['Test'], data['Validation']), ())

source = ColumnDataSource(data=dict(x=x, counts=counts, color=Spectral6))

p = figure(x_range=FactorRange(*x), plot_height=400, plot_width=800, title="Distribution of images with different category",
           tools="hover, pan, box_zoom, wheel_zoom, reset, save", tooltips= ("@x: @counts"))

p.vbar(x='x', top='counts', width=0.9, color='color', legend_field="x", source=source)

p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_center"
output_notebook()
show(p)
# image preprocessing
train_datagen = ImageDataGenerator(rotation_range = 30,
                                   zoom_range = 0.2,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   horizontal_flip = True,
                                   rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 16
training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (224, 224), 
                                                 batch_size = batch_size, 
                                                 class_mode = "binary")
val_set = val_datagen.flow_from_directory(val_dir,
                                          target_size = (224, 224),
                                          batch_size = batch_size,
                                          class_mode = 'binary')
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = (224, 224),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x
def transition_block(x, reduction, name):
    bn_axis = 3
    if K.image_data_format() == "channels_first":
        bn_axis = 1
    x = BatchNormalization(axis = bn_axis, epsilon=1.001e-5, name = name + '_bn')(x)
    x = Activation('relu', name = name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias = False, 
               name = name + '_conv')(x)
    x = AveragePooling2D(2, strides = 2, name = name + '_pool')(x)
    return x
def conv_block(x, growth_rate, name):
    bn_axis = 3
    if K.image_data_format() == "channels_first":
        bn_axis = 1
    
    x1 = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5,
                            name = name + '_0_bn')(x)
    x1 = Activation('relu', name = name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias = False,
                name = name + '_1_conv')(x1)
    
    x1 = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = name + '_1_bn')(x1)
    x1 = Activation('relu', name = name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding = 'same', use_bias = False,
                name = name + '_2_conv')(x1)
    
    x = Concatenate(axis = bn_axis, name = name + '_concat')([x, x1])
    return x
def DenseNet121(blocks, input_shape = (224, 224, 3), classes = 1):
    
    # Determine proper input shape
    img_input = Input(shape=input_shape)
    bn_axis = 3
    if K.image_data_format() == "channels_first":
        bn_axis = 1

    x = ZeroPadding2D(padding = ((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides = 2, use_bias = False, name = 'conv1/conv')(x)
    x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = 'conv1/bn')(x)
    x = Activation('relu', name = 'conv1/relu')(x)
    x = ZeroPadding2D(padding = ((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides = 2, name = 'pool1')(x)

    x = dense_block(x, blocks[0], name = 'conv2')
    x = transition_block(x, 0.5, name = 'pool2')
    x = dense_block(x, blocks[1], name = 'conv3')
    x = transition_block(x, 0.5, name = 'pool3')
    x = dense_block(x, blocks[2], name = 'conv4')
    x = transition_block(x, 0.5, name = 'pool4')
    x = dense_block(x, blocks[3], name = 'conv5')

    x = BatchNormalization(axis = bn_axis, epsilon = 1.001e-5, name = 'bn')(x)
    x = Activation('relu', name = 'relu')(x)
    
    basemodel = Model(img_input, x, name='basedensenet121')
    basemodel.load_weights("../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")
    
    x = basemodel.output
    x = GlobalAveragePooling2D(name = 'avg_pool')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation = 'sigmoid', name = 'fc1')(x)
    
    model = Model(img_input, x, name='densenet121')
    return model
model = DenseNet121(blocks = [6, 12, 24, 16], input_shape = (224, 224, 3), classes = 1)
model.summary()
# compile the model
opt = Adam(lr = 0.001)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
# this will help in reducing learning rate by factor of 0.1 when accuarcy will not improve
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 2, verbose = 1,
                                            factor = 0.3, min_lr = 0.000001)
class myCallback(Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('val_accuracy') > 0.93):
			print("\nReached 93% accuracy so cancelling training!")
			self.model.stop_training = True

callbacks = myCallback()
H = model.fit_generator(training_set,
                        steps_per_epoch = training_set.samples//batch_size,
                        validation_data = val_set,
                        epochs = 15,
                        validation_steps = val_set.samples//batch_size,
                        callbacks = [reduce_lr, callbacks])
print("Loss of the model is - " , model.evaluate(test_set)[0])
print("Accuracy of the model is - " , model.evaluate(test_set)[1]*100 , "%")
# Extract 9 random images from normal
imageset = glob(test_dir + "*/*")
random_images = [np.random.choice(imageset) for i in range(9)]

print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = cv2.imread(random_images[i])
    orig = img.copy()
    img = cv2.resize(img, (299,299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    prediction = model.predict(img)
    
    if (prediction < 0.5):
        plt.title("Normal", fontdict = {'fontsize' : 30})
    
    else:
        plt.title("Pneumonia", fontdict = {'fontsize' : 30})
    plt.imshow(orig, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()  
# Let's try this one more time
imageset = glob(train_dir + "*/*")
random_images = [np.random.choice(imageset) for i in range(9)]

print('Display Random Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = cv2.imread(random_images[i])
    orig = img.copy()
    img = cv2.resize(img, (299,299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    prediction = model.predict(img)
    
    if (prediction < 0.5):
        plt.title("Normal", fontdict = {'fontsize' : 30})
    
    else:
        plt.title("Pneumonia", fontdict = {'fontsize' : 30})
    plt.imshow(orig, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()  