#importing the necessary libraries
import numpy as np
import keras
import shutil
import os
import random
import matplotlib.pyplot as plt
%matplotlib inline

from keras.preprocessing.image import ImageDataGenerator
#assigning image paths to variables
mask_data = "../input/facemask-dataset/Mask/Mask/" 
no_mask_data = "../input/facemask-dataset/No Mask/No Mask/"
total_mask_images = os.listdir(mask_data)
print("no of mask images:: {}".format(len(total_mask_images)))
total_nonmask_images = os.listdir(no_mask_data)
print("no of non-mask images:: {}".format(len(total_nonmask_images)))
os.makedirs('./train/mask')
os.makedirs('./train/no mask')
os.makedirs('./test/mask')
os.makedirs('./test/no mask')
for images in random.sample(total_mask_images,100):
    shutil.copy(mask_data+images, './train/mask')
for images in random.sample(total_mask_images,30):
    shutil.copy(mask_data+images, './test/mask')
for images in random.sample(total_nonmask_images,100):
    shutil.copy(no_mask_data+images, './train/no mask')
for images in random.sample(total_nonmask_images,30):
    shutil.copy(no_mask_data+images, './test/no mask')
train_batch = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, shear_range=0.2).\
            flow_from_directory('./train', target_size=(224,224), batch_size=32, class_mode = 'categorical')
test_batch = ImageDataGenerator(rescale=1./255).\
            flow_from_directory('./test', target_size = (224,224), batch_size=32, class_mode='categorical')
train_batch.class_indices
class_mask = ['mask', 'no mask']
#import vgg16
from keras.applications.vgg16 import VGG16
#vgg16 accepts image size (224,224) only
IMAZE_SIZE = [224,224]
vgg = VGG16(input_shape=IMAZE_SIZE+[3], weights='imagenet', include_top=False)
vgg.summary()
for layers in vgg.layers:
    layers.trainable = False
vgg.summary()
flatten_layer = keras.layers.Flatten()(vgg.output)
prediction_layer = keras.layers.Dense(2, activation='softmax')(flatten_layer)
model = keras.models.Model(inputs = vgg.input, outputs = prediction_layer)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
r = model.fit_generator(train_batch, validation_data=test_batch, epochs=5, steps_per_epoch=len(train_batch), validation_steps=len(test_batch))
plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()


plt.plot(r.history['accuracy'], label = 'train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
img = image.load_img('../input/facemask-dataset/No Mask/No Mask/No Mask109.jpg', target_size=(224,224))
x=image.img_to_array(img)
x = np.expand_dims(x,0)
y = preprocess_input(x)
pred = class_mask[np.argmax(model.predict(y))]
print(pred)
plt.imshow(img)
img = image.load_img('../input/facemask-dataset/Mask/Mask/Mask214.jpeg', target_size=(224,224))
x=image.img_to_array(img)
x = np.expand_dims(x,0)
y = preprocess_input(x)
pred = class_mask[np.argmax(model.predict(y))]
print(pred)
plt.imshow(img)