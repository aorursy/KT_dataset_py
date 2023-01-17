from IPython.display import YouTubeVideo,HTML
YouTubeVideo("IAQp2Zuqevc", width=800, height=500)
!pip install -q efficientnet
# importing libraries
import os
from glob import glob
import pandas as pd
import numpy as np
import numpy as np 
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# nn
from keras.layers.core import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import efficientnet.tfkeras as efn
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint

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
print("Number of images in Trian is {}".format(len(glob(train_dir + "*/*"))))
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
# define architecture

baseModel = efn.EfficientNetB5(weights = "imagenet", include_top = False, input_shape = (299, 299, 3))
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation='sigmoid', name = "efficientnet_dense")(headModel)

model = Model(inputs = baseModel.input, outputs = headModel, name = "EfficientNetB5")

model.trainable = True
model.summary()
# image preprocessing
train_datagen = ImageDataGenerator(rotation_range = 30,
                                   zoom_range = 0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip = True,
                                   rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 16
training_set = train_datagen.flow_from_directory(train_dir, 
                                                 target_size = (299, 299), 
                                                 batch_size = batch_size, 
                                                 class_mode = "binary")
val_set = val_datagen.flow_from_directory(val_dir,
                                          target_size = (299, 299),
                                          batch_size = batch_size,
                                          class_mode = 'binary')
test_set = test_datagen.flow_from_directory(test_dir,
                                          target_size = (299, 299),
                                          batch_size = batch_size,
                                          class_mode = 'binary')
# compile the model
opt = Adam(lr = 0.001)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
# this will help in reducing learning rate by factor of 0.1 when accuarcy will not improve
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 2, verbose = 1,
                                            factor = 0.1, min_lr = 0.000001)
# define criteria for stopping. we will stop training if validation accuracy got reached 98%
class myCallback(Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('val_accuracy') > 0.98):
			print("\nReached 98% accuracy so cancelling training!")
			self.model.stop_training = True

callbacks = myCallback()
checkpoint_filepath = 'xray_model.h5'
model_checkpoint_callback = ModelCheckpoint(filepath = checkpoint_filepath,
                                            save_weights_only = True,
                                            monitor = 'val_acc',
                                            mode = 'max',
                                            save_best_only = True)
H = model.fit_generator(training_set,
                        steps_per_epoch = training_set.samples//batch_size,
                        validation_data = val_set,
                        epochs = 10,
                        validation_steps = val_set.samples,
                        callbacks = [reduce_lr, callbacks, model_checkpoint_callback])
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