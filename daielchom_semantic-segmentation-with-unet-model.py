from skimage import io
import os
import pandas as pd
import numpy as np
import pydicom
from pydicom.data import get_testdata_files
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
PATH = "../input"
print(os.listdir(PATH))
pd.read_csv(PATH+"/stage_1_detailed_class_info.csv").head()
pd.read_csv(PATH+"/stage_1_sample_submission.csv").head()
pd.read_csv(PATH+"/stage_1_train_labels.csv").head()
filename = pydicom.read_file(PATH+"/stage_1_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm")
filename.pixel_array
io.imshow(filename.pixel_array)
train = pd.read_csv(PATH+"/stage_1_train_labels.csv")

test = pd.read_csv(PATH+"/stage_1_sample_submission.csv")
train[train["patientId"] == "00436515-870c-4b36-a041-de91049b9ab4"]
im = np.array(filename.pixel_array, dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect_1 = patches.Rectangle((264,152),213,379,linewidth=1,edgecolor='r',facecolor='none')
rect_2 = patches.Rectangle((562,152),256,453,linewidth=1,edgecolor='r',facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect_1)
ax.add_patch(rect_2)


plt.show()
# Function create by soply on GithubGist
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
plt.show()
filename_2 = pydicom.read_file(PATH+"/stage_1_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm")
show_images([filename_2.pixel_array,filename.pixel_array], titles = ["Not pneumonia", "pneumonia"])
X_train, y_train, train_names = [], [], []
IMG_SIZE = 1024
NEW_SIZE = 128
N = IMG_SIZE/NEW_SIZE
for i in np.unique(train["patientId"]):
    data = train[train["patientId"]== i]
    image = data["patientId"].values[0]
    new_image = Image.fromarray(pydicom.read_file(PATH+"/stage_1_train_images/"+image+".dcm").pixel_array).resize((NEW_SIZE,NEW_SIZE), Image.ANTIALIAS)
    
    X_train.append(np.array(new_image).reshape((NEW_SIZE,NEW_SIZE,1)))
    train_names.append(data["patientId"].values[0])
    if 0 not in data["Target"].values:
        
        new = np.zeros((NEW_SIZE,NEW_SIZE))
        
        for j in range(data.shape[0]):
            x = int(data["x"].values[j]/N)
            y = int(data["y"].values[j]/N)
            width = int(data["width"].values[j]/N)
            height = int(data["height"].values[j]/N)
            new[y:y+height,x:x+width] = 1
        
        y_train.append(new.reshape((NEW_SIZE,NEW_SIZE,1)))
    else:
        new = np.zeros((NEW_SIZE,NEW_SIZE))
        y_train.append(new.reshape((NEW_SIZE,NEW_SIZE,1)))
X_train[9]
X_train[9].shape
show_images([X_train[71].reshape((256,256)),y_train[71].reshape((256,256))], titles=["Image","Ground Truht"])
filename = pydicom.read_file(PATH+"/stage_1_train_images/"+train_names[71]+".dcm")
train[train["patientId"] == train_names[71]]
im = np.array(filename.pixel_array, dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect_1 = patches.Rectangle((698,288),226,311,linewidth=1,edgecolor='r',facecolor='none')
rect_2 = patches.Rectangle((326,212),181,275,linewidth=1,edgecolor='r',facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect_1)
ax.add_patch(rect_2)


plt.show()
show_images([X_train[71].reshape((256,256)),y_train[71].reshape((256,256))], titles=["Image","Ground Truht"])

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
inputs = Input((NEW_SIZE, NEW_SIZE, 1))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(np.array(X_train), np.array(y_train), validation_split=0.1, batch_size=2, epochs=10, callbacks=[earlystopper, checkpointer])
### 2.4 Load Model
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})