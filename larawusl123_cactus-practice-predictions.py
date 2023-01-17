!pip install tensorflow-gpu==2.0.0-beta0

!pip install pandas
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

print(os.listdir("../input/cactus-practise"))
import tensorflow as tf

from tensorflow import keras
def focal_loss_fn(gamma=2.,alpha=0.25):

    EPSILON = 1e-6

    

    def ce(y_true,y_pred,weights=None):

        

        mask = y_pred < EPSILON

        true_vals = tf.fill(tf.shape(y_pred),EPSILON)

        y_pred = tf.where(mask,true_vals,y_pred)

        ce = y_true*(-tf.math.log(y_pred)) + (1-y_true)*(-tf.math.log(1-y_pred))

        

        if weights is not None:

            ce = ce*weights

            

        ce_loss = tf.reduce_mean(ce+EPSILON)

        return ce_loss    

        

    def focal_loss_fixed(y_true,y_pred):

        t = y_true

        p = y_pred

        

        pt = p*t + (1-p)*(1-t)

        w = alpha*t + (1-alpha)*(1-t)

        w = tf.pow((1-pt),gamma)

        

        fl = ce(y_true,y_pred,w)

        

        return fl

    return focal_loss_fixed
model_path = "../input/cactus-practise/c_best_val_weights_resnet50_model.h5"

# model = keras.models.load_model(model_path,custom_objects={"focal_loss_fixed":focal_loss_fn()})

model = keras.models.load_model(model_path)

model.summary()
!pip install imutils
test_imgs_dir = "../input/aerial-cactus-identification/test/test"

# os.listdir(test_imgs_dir)
from imutils import paths



img_paths = list(paths.list_images(test_imgs_dir))

print(len(img_paths))
import cv2
img_size = 96

images = []

img_names = [] 

for img_path in img_paths:

    

    img_name = img_path.split(os.path.sep)[-1]

    img = cv2.imread(img_path)

    img = cv2.resize(img,(img_size,img_size))

    img = keras.preprocessing.image.img_to_array(img)

    

    images.append(img)

    img_names.append(img_name)



    

images = np.array(images,dtype='float') / 255.

print(len(img_names),len(images))

print(np.max(images),np.min(images))
images.shape
predictions = model.predict(images)
# predictions = np.round(predictions)
predictions
img_names = np.array(img_names)

img_names = img_names.reshape((-1,1))

img_names.shape
predictions = predictions.reshape((-1,1))

predictions.shape
solution = np.concatenate((img_names,predictions),axis=1)

solution.shape
submit_csv = pd.DataFrame(solution,columns=["id","has_cactus"])

submit_csv.to_csv("submission_resnet50.csv",index=False)