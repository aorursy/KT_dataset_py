#import necessary packages

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report,confusion_matrix 

from sklearn.metrics import cohen_kappa_score

from imblearn.over_sampling import SMOTE

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm



#Ensures consistency across runs

from numpy.random import seed

seed(1)



#Imports to view data

import cv2

from glob import glob

import matplotlib.pyplot as plt

from numpy import floor

import random



#others

import os

print(os.listdir('../input/'))
train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')

print('Size of train dataset',train_df.shape)

print('Size of test dataset',test_df.shape)
display(train_df.head(6))
train_labels=train_df['diagnosis']

display(train_labels.head())
display(test_df.head(6))
#Target classes

target_classes=['No DR','Mild','Moderate','Sever','Proliferative DR']
#Histo plot for trian data

from sklearn.utils import shuffle

id_code=train_df['id_code']

diagnosis=train_df['diagnosis']

id_code,diagnosis=shuffle(id_code,diagnosis,random_state=42)

plt.figure(figsize=(15,8))

plt.xlabel('diagnosis')

plt.ylabel('count')

diagnosis.hist()
def read_image(path,id_code,size):

    img_path=os.path.join(path,id_code+'.png')

    image=cv2.imread(img_path)

    image=cv2.resize(image,(size,size))

    #Normalizing pixel data (0-255)

    image=image.reshape(size,size,3).astype('float32')/255  

    return image
#Build train images data as a numpy array

train_images=[]

train_images.append(train_df['id_code'].apply(lambda x:read_image

                                              ('../input/train_images',x,128)))

df_train=np.array(train_images)

df_train=df_train.reshape(df_train.shape[1],128,128,3).astype('float32')

print(df_train.shape)
#Build test images data as a numpy array

test_df=pd.read_csv('../input/sample_submission.csv')

test_images=[]

test_images.append(test_df['id_code'].apply(lambda x:read_image

                                              ('../input/test_images',x,128)))

X_test=np.array(test_images)

X_test=X_test.reshape(X_test.shape[1],128,128,3).astype('float32')

print(X_test.shape)
#convert 3d array to 1d array

df_train_2d = len(df_train)

df_train= df_train.reshape(df_train_2d,-1)
#Before sampling

print('Before sampling...')

print('Size of the train dataset:',len(df_train))

print("Before sampling,counts of label '0':{}".format(sum(train_labels==0)))

print("Before sampling,counts of label '1':{}".format(sum(train_labels==1)))

print("Before sampling,counts of label '2':{}".format(sum(train_labels==2)))

print("Before sampling,counts of label '3':{}".format(sum(train_labels==3)))

print("Before sampling,counts of label '4':{}".format(sum(train_labels==4)))

#Apply SMOTE technique

sm=SMOTE(random_state=42,k_neighbors=3)

X_train_res,Y_train_res=sm.fit_sample(df_train,train_labels.ravel())



#After sampling

print('After sampling...')

print('Size of the train dataset:',len(X_train_res))

print("Before sampling,counts of label '0':{}".format(sum(Y_train_res==0)))

print("Before sampling,counts of label '1':{}".format(sum(Y_train_res==1)))

print("Before sampling,counts of label '2':{}".format(sum(Y_train_res==2)))

print("Before sampling,counts of label '3':{}".format(sum(Y_train_res==3)))

print("Before sampling,counts of label '4':{}".format(sum(Y_train_res==4)))
from sklearn.model_selection import train_test_split

X_train,X_valid,Y_train,Y_valid=train_test_split(X_train_res,Y_train_res,

                                                 test_size=0.1,

                                                 random_state=42)

print(len(X_train),len(X_valid))

print(len(Y_train),len(Y_valid))
#convert 3d array to 1d array

X_test_2d = len(X_test)

X_test= X_test.reshape(X_test_2d,-1)

print(X_train.shape,X_valid.shape,X_test.shape)
print('Evaluating the KNN Classifier...')

model=KNeighborsClassifier(n_neighbors=3,n_jobs=-1)

model.fit(X_train,Y_train)

Y_pred=model.predict(X_valid)

classification_report=classification_report(Y_valid,Y_pred,target_names=target_classes)

display(classification_report)

confusion_matrix=confusion_matrix(Y_valid,Y_pred)

display(confusion_matrix)

kappa_score=cohen_kappa_score(Y_valid,Y_pred,weights='quadratic')

print('Quadratic Kappa Score:')

display(kappa_score)
#prediction on unseen data (test data)

KNN=pd.read_csv('../input/sample_submission.csv')

Y_predict=model.predict(X_test)

KNN['diagnosis']=Y_predict

KNN.to_csv('submission.csv',index=False)