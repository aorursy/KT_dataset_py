import numpy as np 

import pandas as pd 
import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import cv2

import sklearn

import seaborn as sb



from skimage.color import rgb2gray

from skimage.filters import laplace, sobel, roberts
s_path ='../input/blur-dataset/sharp/'

df_path='../input/blur-dataset/defocused_blurred/'

mot_path ='../input/blur-dataset/motion_blurred/'



img_paths = ['../input/blur-dataset/sharp/89_IPHONE-6S_S.jpeg','../input/blur-dataset/defocused_blurred/89_IPHONE-6S_F.jpeg','../input/blur-dataset/motion_blurred/89_IPHONE-6S_M.jpeg']

 

def show_images(path):

    plt.figure(figsize=(10,10))

    for i in range(len(path)):

        x=plt.imread(path[i])

        plt.subplot(1, 3, i+1)

        plt.imshow(x)

        plt.xticks([])

        plt.yticks([])

    plt.tight_layout()

    plt.show()

show_images(img_paths)
sharp_images = os.listdir(s_path)

defocused = os.listdir(df_path)

motion_blurred = os.listdir(mot_path)
def get_data(path,images):

    features=[]

    for img in images:

        feature=[]

        gray = cv2.imread(path+img,0)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        feature.append(laplacian_var)

        features.append(feature)

    return features
sharp_features = get_data(s_path,sharp_images)

defocused_features = get_data(df_path,defocused)

motion_blur_features = get_data(mot_path,motion_blurred)

sharp_df = pd.DataFrame(sharp_features)

print(sharp_df[:10])
defocused_df = pd.DataFrame(defocused_features)

defocused_df.head()
motion_df = pd.DataFrame(motion_blur_features)

motion_df.head()
from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, classification_report



images=pd.DataFrame()

images = images.append(sharp_df)

images = images.append(defocused_df)

images = images.append(motion_df)

all_features = np.array(images)

y_f = np.concatenate((np.ones((sharp_df.shape[0], )), np.zeros((defocused_df.shape[0]+motion_df.shape[0], ))), axis=0)



x_train,x_valid,y_train,y_valid = train_test_split(all_features,y_f,test_size=0.33,stratify=y_f)



svm_model = svm.LinearSVC(dual=False,random_state=89)

svm_model.fit(x_train,y_train)

pred =svm_model.predict(x_valid)

print('Accuracy:',accuracy_score(y_valid,pred))
import sklearn

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, classification_report



images=pd.DataFrame()

images = images.append(sharp_df)

images = images.append(defocused_df)

images = images.append(motion_df)

all_features = np.array(images)

y_f = np.concatenate((np.ones((sharp_df.shape[0], )), np.zeros((defocused_df.shape[0]+motion_df.shape[0], ))), axis=0)



x_train,x_valid,y_train,y_valid = train_test_split(all_features,y_f,test_size=0.33,stratify=y_f)



reg_model = sklearn.linear_model.LogisticRegression()

reg_model.fit(x_train,y_train)

pred =reg_model.predict(x_valid)

print('Accuracy:',accuracy_score(y_valid,pred))
import pickle

filename='Regressor_model.sav'

pickle.dump(reg_model, open(filename, 'wb'))
!ls
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, classification_report



images=pd.DataFrame()

images = images.append(sharp_df)

images = images.append(defocused_df)

images = images.append(motion_df)

all_features = np.array(images)

y_f = np.concatenate((np.ones((sharp_df.shape[0], )), np.zeros((defocused_df.shape[0]+motion_df.shape[0], ))), axis=0)



x_train,x_valid,y_train,y_valid = train_test_split(all_features,y_f,test_size=0.33,stratify=y_f)



dt_model = DecisionTreeClassifier()

dt_model.fit(x_train,y_train)

pred =dt_model.predict(x_valid)

print('Accuracy:',accuracy_score(y_valid,pred))


pred=reg_model.predict([[450]])

print(pred)
