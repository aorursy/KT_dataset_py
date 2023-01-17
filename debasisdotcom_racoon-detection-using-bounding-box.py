import numpy as np

import pandas as pd

from keras import Model

from keras.applications.mobilenet import MobileNet, preprocess_input

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

from keras.layers import Conv2D, Reshape

from keras.utils import Sequence

from keras.backend import epsilon

import tensorflow as tf



from PIL import Image



import os

import random



import matplotlib.pyplot as plt

import matplotlib.patches as patches

from PIL import Image

import numpy as np

import cv2



np.random.seed(1)
train = pd.read_csv("../input/racoon-detection/train_labels_.csv")
train.head()
train.shape
IMAGE_SIZE = 128
coords=train[["width","height","xmin","ymin","xmax","ymax"]]



coords["xmin"] = coords["xmin"] *IMAGE_SIZE/coords["width"]

coords["xmax"] = coords["xmax"]*IMAGE_SIZE /coords["width"]

coords["ymin"] = coords["ymin"] *IMAGE_SIZE/coords["height"]

coords["ymax"] = coords["ymax"] *IMAGE_SIZE/coords["height"]



coords.drop(["width","height"],axis =1,inplace=True)

coords.head()
paths = train["filename"]

len(paths)
images = "../input/racoon-detection/Racoon Images/images/"



batch_images = np.zeros((len(paths), IMAGE_SIZE, IMAGE_SIZE,3), dtype=np.float32)



for i, f in enumerate(paths):

  #print(f)

  img = Image.open(images+f)

  img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

  img = img.convert('RGB')

  batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
ALPHA = 1.0



model = MobileNet(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, alpha=ALPHA)
for layers in model.layers:

  layers.trainable = False



x = model.layers[-1].output

x = Conv2D(4, kernel_size = 4, name="coords")(x)

x = Reshape((4,))(x)



model = Model(inputs = model.inputs, outputs = x)
def loss(gt,pred):

    intersections = 0

    unions = 0

    diff_width = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])

    diff_height = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])

    intersection = diff_width * diff_height

    

    # Compute union

    area_gt = gt[:,2] * gt[:,3]

    area_pred = pred[:,2] * pred[:,3]

    union = area_gt + area_pred - intersection



#     Compute intersection and union over multiple boxes

    for j, _ in enumerate(union):

        if union[j] > 0 and intersection[j] > 0 and union[j] >= intersection[j]:

            intersections += intersection[j]

            unions += union[j]



    # Compute IOU. Use epsilon to prevent division by zero

    iou = np.round(intersections / (unions + epsilon()), 4)

    iou = iou.astype(np.float32)

    return iou



def IoU(y_true, y_pred):

    iou = tf.py_function(loss, [y_true, y_pred], tf.float32)

    return iou
gt = coords



PATIENCE=10



model.compile(optimizer = "Adam", loss = "mse", metrics = [IoU])



stop = EarlyStopping(monitor='val_iou', patience=PATIENCE, mode="max" )



reduce_lr = ReduceLROnPlateau(monitor='val_iou',factor=0.2,patience=PATIENCE, min_lr=1e-7, verbose=1, mode="max" )



model.fit(batch_images, gt, epochs=100,callbacks=[stop,reduce_lr], verbose = 2)
test_img = random.choice(paths)
filename = images+ test_img

unscaled = cv2.imread(filename)
image_height, image_width, _ = unscaled.shape

image = cv2.resize(unscaled,(IMAGE_SIZE,IMAGE_SIZE))

feat_scaled = preprocess_input(np.array(image, dtype=np.float32))
region = model.predict(x = np.array([feat_scaled]))[0]
x0 = int(region[0] * image_width / IMAGE_SIZE) 

y0 = int(region[1] * image_height / IMAGE_SIZE)



x1 = int((region[2]) * image_width / IMAGE_SIZE)

y1 = int((region[3]) * image_height / IMAGE_SIZE)
# Create figure and axes

fig,ax = plt.subplots(1)



# Display the image

ax.imshow(unscaled)



# Create a Rectangle patch

rect = patches.Rectangle((x0, y0), (x1 - x0) , (y1 - y0) , linewidth=2, edgecolor='r', facecolor='none')



# Add the patch to the Axes

ax.add_patch(rect)



plt.show()