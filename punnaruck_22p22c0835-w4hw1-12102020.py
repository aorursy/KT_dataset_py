# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json

import math

import os

import time



import cv2

from PIL import Image

import numpy as np

import keras

from keras import layers

from keras.applications import DenseNet121, DenseNet201

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

from keras.models import Model

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, confusion_matrix, classification_report

from sklearn.utils import class_weight, shuffle

from sklearn.model_selection import KFold

import scipy

import tensorflow as tf

from tqdm import tqdm

from keras import backend as K

from keras.models import Model, save_model,load_model

import tensorflow as tf



import matplotlib.image as mpimg

from skimage.morphology import convex_hull_image

from skimage.util import invert



plt.gray()



import gc

gc.collect()
IMG_SIZE = 224

BATCH_SIZE = 16
pdTrainMap = pd.read_csv("/kaggle/input/thai-mnist-classification/mnist.train.map.csv")
pdTrainMap.head()
pdTrainMap.groupby("category").count()
plt.figure(figsize=(22, 22))



number = 10

img_list = []

df_map = pdTrainMap

training_dir = "/kaggle/input/thai-mnist-classification/train"

for i in range(number):

    temp = list(df_map[df_map['category'] == i]['id'][:10])

    img_list = img_list + temp



for index, file in enumerate(img_list):

    path = os.path.join(training_dir,file)

    plt.subplot(number,len(img_list)/number,index+1)

    img = mpimg.imread(path)

    plt.axis('off')

    plt.imshow(img)

    

gc.collect()
def convex_crop(img,pad=20):

    convex = convex_hull_image(img)

    r,c = np.where(convex)

    diff_r = max(r) - min(r)

    diff_c = max(c) - min(c)

    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):

        pad = pad - 1

    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
fig, [ax1,ax2] = plt.subplots(1, 2)



path1 = f'/kaggle/input/thai-mnist-classification/train/b8fd3385-9403-48a4-9d9e-74bde635e688.png'

img1 = cv2.imread(path1)

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

temp_img = invert(img1)

ax1.imshow(temp_img)



cvh =  convex_crop(temp_img)

ax2.imshow(cvh)

gc.collect()
fig, [ax1,ax2, ax3] = plt.subplots(1, 3)



path1 = f'/kaggle/input/thai-mnist-classification/train/b8fd3385-9403-48a4-9d9e-74bde635e688.png'

img1 = cv2.imread(path1)

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

temp_img = invert(img1)

ax1.imshow(temp_img)



cvh =  convex_crop(temp_img)

ax2.imshow(cvh)



kernel = np.ones((5,5),np.uint8)

dilation = cv2.dilate(cvh,kernel,iterations = 1)

ax3.imshow(dilation)

gc.collect()
def preprocessing_image(image_path, IMG_SIZE=224, pad=30):

    img1 = cv2.imread(image_path)

    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    img = invert(img1)

    

    convex = convex_hull_image(img)

    r,c = np.where(convex)

    diff_r = max(r) - min(r)

    diff_c = max(c) - min(c)

    

    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):

        pad = pad - 1

    

    image_cropped = img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]

    image_resize = cv2.resize(image_cropped, (IMG_SIZE, IMG_SIZE))

    

    kernel = np.ones((5,5),np.uint8)

    image_resize = cv2.dilate(image_resize,kernel,iterations = 1)

#     print("image_resize", image_resize.shape)

#     image_resize = invert(image_resize)

    image_resize = cv2.cvtColor(image_resize,cv2.COLOR_GRAY2BGR)

#     image_resize = cv2.threshold(image_resize, 25, 255, cv2.THRESH_BINARY_INV)[1]

#     image_resize = image_resize/255.

#     print("image_resize", image_resize.shape)



#     del kernel

#     del image_cropped

#     del convex

#     del img

#     del img1

    

#     gc.collect()

    return image_resize, diff_r, diff_c
fig, [ax1,ax2] = plt.subplots(1, 2)



path1 = f'/kaggle/input/thai-mnist-classification/train/b8fd3385-9403-48a4-9d9e-74bde635e688.png'

img1 = cv2.imread(path1)

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

temp_img = invert(img1)

ax1.imshow(temp_img)



tic = time.perf_counter()

cvh,r,c =  preprocessing_image(path1)

toc = time.perf_counter()

print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")



print(cvh.shape)

print(cvh[0][0])

ax2.imshow(cvh)



gc.collect()
plt.figure(figsize=(22, 22))



number = 10

img_list = []

df_map = pdTrainMap

training_dir = "/kaggle/input/thai-mnist-classification/train"

for i in range(number):

    temp = list(df_map[df_map['category'] == i]['id'][:10])

    img_list = img_list + temp



for index, file in enumerate(img_list):

    path = os.path.join(training_dir,file)

    plt.subplot(number,len(img_list)/number,index+1)

    img,r,c = preprocessing_image(path)

    plt.axis('off')

    plt.imshow(img)
gc.collect()
x_train = []

y_train_temp = []



for index, row in pdTrainMap.iterrows():

    image_id = row['id']

    image_path = f'/kaggle/input/thai-mnist-classification/train/{image_id}'

    image_processed, diff_r, diff_c = preprocessing_image(image_path)

    

    if diff_r >= 100 and diff_c >= 100:

        x_train.append(image_processed)

        y_train_temp.append(row['category'])

    

#     del image_processed

#     gc.collect()



x_train = np.array(x_train)

y_train = pd.get_dummies(y_train_temp).values
print(len(x_train))

print(len(y_train))
pdTrainMap.count()
pdTrainRule = pd.read_csv("/kaggle/input/thai-mnist-classification/train.rules.csv")
pdTrainRule.head()
dfF1 = pdTrainMap.rename(columns={"id": "feature1", "category": "label1"})

dfF2 = pdTrainMap.rename(columns={"id": "feature2", "category": "label2"})

dfF3 = pdTrainMap.rename(columns={"id": "feature3", "category": "label3"})
pdTrainRuleLabel = pd.merge(pdTrainRule, dfF1, on=["feature1"], how="left")

pdTrainRuleLabel = pd.merge(pdTrainRuleLabel, dfF2, on=["feature2"], how="left")

pdTrainRuleLabel = pd.merge(pdTrainRuleLabel, dfF3, on=["feature3"], how="left")
del dfF1

del dfF2

del dfF3
pdTrainRuleLabel.head()
pdTrainDetails = pdTrainRuleLabel[['id', 'label1', 'label2', 'label3', 'predict']]
def build_model_functional():

    densenet = DenseNet201(

        weights='imagenet',

        include_top=False,

        input_shape=(IMG_SIZE,IMG_SIZE,3)

    )



    base_model = densenet

    GAP_layer = layers.GlobalAveragePooling2D()

    drop_layer = layers.Dropout(0.6)

    dense_layer = layers.Dense(10, activation='softmax', name='final_output')

    

    x = GAP_layer(base_model.layers[-1].output)

    x = drop_layer(x)

    final_output = dense_layer(x)

    model = Model(base_model.layers[0].input, final_output)

    

    return model
acc_per_fold = []

loss_per_fold = []

kf = KFold(n_splits = 5)

fold_no = 1

modelOne = build_model_functional() # with pretrained weights, and layers we want

modelOne.compile(

    loss='categorical_crossentropy',

    optimizer=Adam(lr=0.00005),

    metrics=['accuracy']

)



# x_train = np.array(x_train_tmp)

for train, test in kf.split(x_train, y_train):



    # Generate a print

    print('------------------------------------------------------------------------')

    print(f'Training for fold {fold_no} ...')



    # Fit data to model

    history = modelOne.fit(x_train[train], y_train[train],

              batch_size=BATCH_SIZE,

              epochs=5,

              validation_data=(x_train[test], y_train[test]),

              verbose=1)

    

    # Generate generalization metrics

    scores = modelOne.evaluate(x_train[test], y_train[test], verbose=0)

    print(f'Score for fold {fold_no}: {modelOne.metrics_names[0]} of {scores[0]}; {modelOne.metrics_names[1]} of {scores[1]*100}%')

    acc_per_fold.append(scores[1] * 100)

    loss_per_fold.append(scores[0])



    # Increase fold number

    fold_no = fold_no + 1

    

#     del train

#     del test
# == Provide average scores ==

print('------------------------------------------------------------------------')

print('Score per fold')

for i in range(0, len(acc_per_fold)):

    print('------------------------------------------------------------------------')

    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')

print('------------------------------------------------------------------------')

print('Average scores for all folds:')

print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')

print(f'> Loss: {np.mean(loss_per_fold)}')

print('------------------------------------------------------------------------')
import gc

gc.collect()
# del x_train

# del y_train

# del kf
pdTestRule = pd.read_csv("/kaggle/input/thai-mnist-classification/test.rules.csv")
pdTestRule
def numExpress(f1,f2,f3):

    if np.isnan(f1):

        return f2+f3

    elif f1 == 0:

        return f2*f3

    elif f1 == 1:

        return abs(f2-f3)

    elif f1 == 2:

        return (f2+f3)*abs(f2-f3)

    elif f1 == 3:

        return abs((f3*(f3 +1) - f2*(f2-1))/2)

    elif f1 == 4:

        return 50+(f2-f3)

    elif f1 == 5:

        return min(f2,f3)

    elif f1 == 6:

        return max(f2,f3)

    elif f1 == 7:

        return ((f2*f3)%9)*11

    elif f1 == 8:

        p = ((f2**2)+1)*(f2) +(f3)*(f3+1)

        if p > 99:

            return p%99

        else:

            return p

    elif f1 == 9:

        return 50+f2
import os



listTest = []

for dirname, _, filenames in os.walk('/kaggle/input/thai-mnist-classification/test'):

    for filename in filenames:

#         print(os.path.join(dirname, filename))

        fullPath = os.path.join(dirname, filename)

        listTest.append({"id": filename, "id_path": fullPath})

        

pdImageTest = pd.DataFrame(listTest)
pdImageTest
def prepareValSet(pdValidate):

    N = pdValidate.shape[0]

    x_val_set = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)



    for i, image_id in enumerate(tqdm(pdValidate['id_path'])):

        x_val_set[i, :, :, :] = preprocessing_image(image_id)[0]

#         gc.collect()

    

    return x_val_set
x_validate = prepareValSet(pdImageTest)
predictions = modelOne.predict(x_validate).argmax(axis=-1)

predictions
pdImageTest['category'] = predictions
dfF1 = pdImageTest[['id', 'category']].rename(columns={"id": "feature1", "category": "label1"})

dfF2 = pdImageTest[['id', 'category']].rename(columns={"id": "feature2", "category": "label2"})

dfF3 = pdImageTest[['id', 'category']].rename(columns={"id": "feature3", "category": "label3"})
pdTestRule = pd.merge(pdTestRule, dfF1, on="feature1", how="left")

pdTestRule = pd.merge(pdTestRule, dfF2, on="feature2", how="left")

pdTestRule = pd.merge(pdTestRule, dfF3, on="feature3", how="left")
pdTestRule['predict'] = pdTestRule.apply(lambda x: numExpress(x['label1'],x['label2'], x['label3']), axis=1)
pdTestRule[['label1', 'label2', 'label3','predict']].head(100)
pdTestRule.describe()
pdTestRule[['label1', 'label2', 'label3','predict']][pdTestRule.predict > 99].head(100)
pdTestRule
pdTestRule[['id', 'predict']].to_csv("submit.csv", index=False)