import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input/leapgestrecog/leapGestRecog"))
from PIL import Image

import matplotlib.image as mpimg 

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import IPython.display

path='../input/leapgestrecog/leapGestRecog'

folders=os.listdir(path)

folders=set(folders)



#import codecs

#import json





different_classes=os.listdir(path+'/'+'00')

different_classes=set(different_classes)









print("The different classes that exist in this dataset are:")

print(different_classes)
x=[]

z=[]

y=[]#converting the image to black and white

threshold=200

import cv2





for i in folders:

    print('***',i,'***')

    subject=path+'/'+i

    subdir=os.listdir(subject)

    subdir=set(subdir)

    for j in subdir:

        #print(j)

        images=os.listdir(subject+'/'+j)

        for k in images:

            results=dict()

            results['y']=j.split('_')[0]

            img = cv2.imread(subject+'/'+j+'/'+k,0)

            img=cv2.resize(img,(int(160),int(60)))

            

            ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            imgD=np.asarray(img,dtype=np.float64)

            z.append(imgD)

            imgf=np.asarray(imgf,dtype=np.float64)

            x.append(imgf)

            y.append(int(j.split('_')[0]))

            results['x']=imgf
l = []

list_names = []

for i in range(10):

    l.append(0)

for i in range(len(x)):

    if(l[y[i] - 1] == 0):

        l[y[i] - 1] = i

        if(len(np.unique(l)) == 10):

            break

for i in range(len(l)):

    %matplotlib inline

    print("Class Label: " + str(i + 1))

    plt.imshow(np.asarray(z[l[i]]), cmap  =cm.gray)

    plt.show()

    plt.imshow(np.asarray(x[l[i]]), cmap = cm.gray)     

    plt.show()
x=np.array(x)

y=np.array(y)

y = y.reshape(len(x), 1)

print(x.shape)

print(y.shape)

print(max(y),min(y))
x_data = x.reshape((len(x), 60, 160, 1))



x_data/=255

x_data=list(x_data)

for i in range(len(x_data)):

    x_data[i]=x_data[i].flatten()
len(x_data)
from sklearn.decomposition import PCA

pca = PCA(n_components=20)

x_data=np.array(x_data)

print("Before PCA",x_data.shape)
x_data=pca.fit_transform(x_data)

print(pca.explained_variance_ratio_)  

print(pca.singular_values_)  



print('___________________')

print("After PCA",x_data.shape)
from sklearn.model_selection import train_test_split

x_train,x_further,y_train,y_further = train_test_split(x_data,y,test_size = 0.2)

x_train,x_valid,y_train, y_valid = train_test_split(x_train,y_train,test_size = 0.5)
from sklearn.preprocessing import MinMaxScaler  

scaler = MinMaxScaler()  

#We use the MinMaxScaler for Naive Bayes because some of the classifiers require non-negative input.

#We normalize the values between 0 and 1 in this case

scaler.fit(x_train)



X_train = scaler.transform(x_train)  

X_valid = scaler.transform(x_valid)

X_test = scaler.transform(x_further)  
from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import ComplementNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB



classifiers = ['bernoulli', 'complement', 'gaussian', 'multinomial']

errors = []



bnb = BernoulliNB()

bnb.fit(X_train, y_train)

error = 1. - bnb.score(X_valid, y_valid)

errors.append(error)



compnb = ComplementNB()

compnb.fit(X_train, y_train)

error = 1. - compnb.score(X_valid, y_valid)

errors.append(error)



gnb = GaussianNB() 

gnb.fit(X_train, y_train)

error = 1. - gnb.score(X_valid, y_valid)

errors.append(error)



mnb = MultinomialNB()

mnb.fit(X_train, y_train)

error = 1. - mnb.score(X_valid, y_valid)

errors.append(error)



plt.plot(classifiers, errors)

plt.title('Classifier vs. Model Error')

plt.xlabel('classifier')

plt.ylabel('error')

plt.xticks(classifiers)

plt.show()



minError = errors.index(min(errors))

bestClassifier = classifiers[minError]



print("Optimal Classifier: {}".format(bestClassifier))
smoothing_values =  [1e-15, 1e-11, 1e-9, 1e-5, 0.001, 0.01, 0.1, 1]

smoothing_error = []



for smoothing in smoothing_values:

    model = GaussianNB(var_smoothing=smoothing)

    model = model.fit(X_train, y_train)

    error = 1. - model.score(X_valid, y_valid)

    smoothing_error.append(error)

    

plt.plot(smoothing_values, smoothing_error)

plt.title('Smoothing vs. Model Error')

plt.xlabel('smoothng')

plt.ylabel('error')

#plt.xticks(smoothing)

plt.show()



minError =smoothing_error.index(min(smoothing_error))

bestSmoothing = smoothing_values[minError]



print("Best Smoothing: {}".format(bestSmoothing))
bestnb = GaussianNB(var_smoothing=bestSmoothing)

bestnb.fit(X_train, y_train)

y_pred_nb = bestnb.predict(X_test)

y_train_score_nb = bestnb.predict(X_train)
from sklearn.metrics import accuracy_score

print("accuracy of the model is:\nTest ", accuracy_score(y_further, y_pred_nb, normalize=True, sample_weight=None))

print('Train',accuracy_score(y_train, y_train_score_nb, normalize=True, sample_weight=None))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



matrix = confusion_matrix(y_further, y_pred_nb, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

precision = precision_score(y_further, y_pred_nb, average = None)

accuracy = accuracy_score(y_further, y_pred_nb, normalize=True, sample_weight=None)

recall = recall_score(y_further, y_pred_nb, average=None)



print("Confusion Matrix:\n", matrix, "\n")

print("Accuracy:", accuracy, "\n")

print("Recall:", recall, "\n")

print("Precision:", precision)