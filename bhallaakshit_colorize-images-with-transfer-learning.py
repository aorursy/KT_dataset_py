import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16

from skimage import io
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
color_path = "../input/human-faces/Humans"
color_images = os.listdir(color_path)

# Randomly view 9 color images
size = 3
images = random.sample(color_images, size*size)
images = np.array(images)
images = images.reshape(size, size)

fig, axs = plt.subplots(size, size, figsize = (15, 15))
for i in range(size):
    for j in range(size):
        img_path = os.path.join(color_path, images[i, j])
        img = io.imread(img_path)
        axs[i, j].imshow(img)
        axs[i, j].set(xticks = [], yticks = [])

fig.tight_layout()
# View 9 black-white images for the same random images
fig, axs = plt.subplots(size, size, figsize = (15, 15))
for i in range(size):
    for j in range(size):
        img_path = os.path.join(color_path, images[i, j])
        img = io.imread(img_path)
        img = rgb2gray(img)
        axs[i, j].imshow(img, cmap = plt.cm.gray)
        axs[i, j].set(xticks = [], yticks = [])

fig.tight_layout()
# Build image data generator 
train_datagen = ImageDataGenerator(
    rescale = 1./255 # Normalization
)

# Obtain all images from directory
batch_size = 1500
target_size = 256
train = train_datagen.flow_from_directory(
    "../input/human-faces", 
    target_size = (target_size, target_size),
    batch_size = batch_size
)
# Convert rgb images to lab
X = []
Y = []
for img in train[0]:
    try:
        lab = rgb2lab(img)
        X.append(lab[:, :, :, 0])
        Y.append(lab[:, :, :, 1:] / 128)
    except:
        print("error")
# Reshape arrays to suit model input
X = np.array(X)
Y = np.array(Y)

X = X.reshape(batch_size, target_size, target_size, -1)
Y = Y.reshape(batch_size, target_size, target_size, -1) 

print(X.shape)
print(Y.shape)
# VGG accepts input of shape (256, 256, 3) so repeat the layer two times 
X = np.repeat(X, 3, axis=3)
# Load the VGG16 model
encoder = VGG16(
    weights = "imagenet",
    include_top = False, 
    input_tensor = Input((256, 256, 3))
)
# print the model summary
encoder.summary()
# Unfreeze the weights in the base model, now these weights will be changed during training
encoder.trainable = True
#Decoder
decoder = Conv2D(512, (3, 3), activation = "relu", padding = "same")(encoder.output)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D(256, (3, 3), activation = "relu", padding = "same")(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D(128, (3, 3), activation = "relu", padding = "same")(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D( 64, (3, 3), activation = "relu", padding = "same")(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2D( 32, (3, 3), activation = "relu", padding = "same")(decoder)
decoder = Conv2D( 16, (3, 3), activation = "relu", padding = "same")(decoder)
decoder = Conv2D(  8, (3, 3), activation = "relu", padding = "same")(decoder)
decoder = Conv2D(  4, (3, 3), activation = "relu", padding = "same")(decoder)
decoder = Conv2D(  2, (3, 3), activation = "tanh", padding = "same")(decoder)
decoder = UpSampling2D((2, 2))(decoder)

# Model
model = Model(inputs = encoder.input, outputs = decoder)
model.summary()
# Compile model
model.compile(
    optimizer = "adam", 
    loss = "mse", 
    metrics = ['accuracy']
)
# Fit the model (mapping input image to output image)
history = model.fit(
    X, Y,
    epochs = 200,
    callbacks = [
        ModelCheckpoint("model_weights.h5")
    ]
)
# Plot loss curve
plt.plot(history.history["loss"])
plt.legend(["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss with epochs")
plt.show()

# Visualizing original, input and output images
fig, axs = plt.subplots(1, 3, figsize = (15, 5))

# Plot original image
original_img = io.imread("../input/andrew-ng/andrew.jpg")
axs[0].imshow(original_img)
axs[0].set(xlabel = "Original Image", xticks = [], yticks = [])

# Plot gray image (input)
img = original_img/255.
img = resize(img, (target_size, target_size, 3))
img = rgb2lab(img)
gray_img = img[:, :, 0]
axs[1].imshow(gray_img, cmap = plt.cm.gray)
axs[1].set(xlabel = "Gray Image (input)", xticks = [], yticks = [])

# Make prediction on the input to get output
gray_img = gray_img.reshape(1, target_size, target_size, -1)
gray_img = np.repeat(gray_img, 3, axis = 3) ###

pred = model.predict(gray_img)
pred = pred.reshape(target_size, target_size, 2)
gray_img = gray_img.reshape(target_size, target_size, 3)

# Plot colorized image (output)
result = np.zeros((target_size, target_size, 3))
result[:, :, 0] = gray_img[:, :, 0]
result[:, :, 1:] = pred*128
result = lab2rgb(result)
axs[2].imshow(result)
axs[2].set(xlabel = "Colorized Image (output)", xticks = [], yticks = [])

