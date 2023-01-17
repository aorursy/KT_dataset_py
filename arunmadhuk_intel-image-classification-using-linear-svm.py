import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

import cv2



from tqdm import tqdm

import random as rn

from random import shuffle  

from zipfile import ZipFile

from PIL import Image



from skimage import feature, color, data

from sklearn.preprocessing import LabelEncoder
# The training data set is in the /Users/macos/Documents/Intel Image Classification/seg_train

trn_img_path = "../input/seg_train/seg_train/"



# The testing data set is in the /Users/macos/Documents/Intel Image Classification/seg_test

tst_img_path = "../input/seg_test/seg_test/"



# Lets create 2 set of arrays for train & testing data's. One for to store the Image data and anther one for label details

X_train =[] # Stores the training image hog data

label_train = [] # Stores the training image label



X_test = [] # Stores the testing image hog data

label_test = [] # Stores the testing image label



scene_label=['Buildings','Forest', 'Glacier','Mountain','Sea','Street']


def hog_data_extractor(jpeg_path):

    jpeg_data = cv2.imread(jpeg_path)

    jpeg_data=cv2.resize(jpeg_data,(150,150)) 

    hog_data = feature.hog(jpeg_data)/255.0

    return hog_data
def jpeg_to_array (scene_type, img_root_path,data_type):

    scene_path = os.path.join(img_root_path,scene_type.lower())

    print('Loading ' + data_type +' images for scene type '+scene_type)

    for img in os.listdir(scene_path):

        img_path = os.path.join(scene_path,img)

        if img_path.endswith('.jpg'):

            if(data_type == 'Training'):

                X_train.append(hog_data_extractor(img_path))

                label_train.append(str(scene_type))

            if(data_type =='Testing'):

                X_test.append(hog_data_extractor(img_path))

                label_test.append(np.array(str(scene_type)))
[jpeg_to_array(scene,trn_img_path,'Training')for scene in scene_label]

len(X_train)

[jpeg_to_array(scene,tst_img_path,'Testing')for scene in scene_label]

len(X_test)
le = LabelEncoder()

y_train = le.fit_transform(label_train)

y_test = le.fit_transform(label_test)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.svm import LinearSVC
lsvc = LinearSVC(random_state=0,tol=1e-5)

lsvc.fit(X_train,y_train)

print('Coef',lsvc.coef_)

print('Intercept',lsvc.intercept_)
# filter all the warnings

import warnings

warnings.filterwarnings('ignore')



# 10-fold cross validation

lsvc_score = lsvc.score(X_test,y_test)

print('Score', lsvc_score)

kfold = KFold(n_splits=10, random_state=9)

cv_results = cross_val_score(lsvc , X_train, y_train, cv=kfold, scoring="accuracy")

print(cv_results)
print(cv_results.mean(), cv_results.std())
def scene_predict(img_path):

    image = cv2.imread(img_path)

    ip_image = Image.open(img_path)

    image = cv2.resize(image,(150,150))

    prd_image_data = hog_data_extractor(img_path)

    scene_predicted = lsvc.predict(prd_image_data.reshape(1, -1))[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6),

                       subplot_kw=dict(xticks=[], yticks=[]))

    ax[0].imshow(ip_image)

    ax[0].set_title('input image')



    ax[1].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    ax[1].set_title('Scene predicted :'+ scene_label[scene_predicted]);
ip_img_folder = '../input/seg_pred/seg_pred/'

ip_img_files = ['222.jpg','121.jpg','88.jpg','398.jpg','839.jpg', '520.jpg']

scene_predicted = [scene_predict(os.path.join(ip_img_folder,img_file))for img_file in ip_img_files]