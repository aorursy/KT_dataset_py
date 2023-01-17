#Essential Imports
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
import os
import pickle
import csv
#Reading and preparing the images
images = []
labels = []
limit=400
img_size=256

main_path='/kaggle/input/chest-xray-pneumonia/chest_xray/train/'
folder_names = []
for entry_name in os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/'):
    entry_path = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/train/', entry_name)
    if os.path.isdir(entry_path):
        folder_names.append(entry_name)
        
print('The Categories are',folder_names)

j=0
for folder in folder_names:
    for filename in os.listdir(os.path.join(main_path,folder)):
        img_path = os.path.join(main_path,folder)
        img = cv2.imread(os.path.join(img_path,filename)) 
        if img is not None:
            img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)   
            img = cv2.resize(img,(img_size, img_size))
            images.append(img)
            if folder == 'NORMAL':
                labels.append(0)
                #print('normal')
            else:
                labels.append(1)
                #print('PNE')
        j=j+1
        if j >= limit:
            j=0
            break
            
images,labels=np.array(images),np.array(labels)
print(images.shape)
#print(labels)
#Printing a random sample
from random import randrange

i = randrange(limit*2)

plt.imshow(images[i],cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#Canny
canny = cv2.Canny(images[i],40,200)


#Sobel
sobelY = cv2.Sobel(images[i],cv2.CV_8UC1,0,1,ksize=5)

preview = [canny,sobelY]
i=0
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(preview[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
#Obtaining Variance of images
kernel = np.ones((3,3),np.uint8)
var_vector = np.empty((limit*2,1))
i = 0
for image in images:
    x, bins = np.histogram(image,bins=255, density=False)
    var_vector[i] = np.var(x)
    i=i+1
#print(var_vector[6])
#Obtaining Mean of images
mean_vector = np.empty((limit*2,1))
i = 0
for image in images:
    x, bins = np.histogram(image,bins=255, density=False)
    mean_vector[i] = np.mean(x)
    i=i+1
#print(mean_vector)
#Obtaining Standard Deviation of images
std_vector = np.empty((limit*2,1))
i = 0
for image in images:
    x, bins = np.histogram(image,bins=255, density=False)
    std_vector[i] = np.std(x)
    i=i+1
#print(std_vector)
#Obtaining Skewness of images
skew_vector = np.empty((limit*2,1))
i = 0
for image in images:
    x, bins = np.histogram(image,bins=255, density=False)
    skew_vector[i] = stats.skew(x)
    i=i+1
#print(skew_vector)
#Obtaining Kurtosis of images
kurto_vector = np.empty((limit*2,1))
i = 0
for image in images:
    x, bins = np.histogram(image,bins=255, density=False)
    kurto_vector[i] = stats.kurtosis(x)
    i=i+1
#print(kurto_vector)
#Obtaining Entropy of images
entropy_vector = np.empty((limit*2,1))
i = 0
for image in images:
    x, bins = np.histogram(image,bins=255, density=False)
    entropy_vector[i] = stats.entropy(x)
    i=i+1
#print(entropy_vector)
#Applying Canny edge detection
canny_vector = np.empty((limit*2,img_size*img_size))
i = 0
for image in images:
    canny = cv2.Canny(image,40,200)
    canny_vector[i] = np.array(canny.flatten())
    i=i+1
#print(canny_vector[1])
#Applying Sobel X
sobelX_vector = np.empty((limit*2,img_size*img_size))
i = 0
for image in images:
    sobelX = cv2.Sobel(image,cv2.CV_8UC1,1,0,ksize=5)
    sobelX_vector[i] = np.array(sobelX.flatten())
    i=i+1
#Applying Sobel Y
sobelY_vector = np.empty((limit*2,img_size*img_size))
i = 0
for image in images:
    sobelY = cv2.Sobel(image,cv2.CV_8UC1,0,1,ksize=5)
    sobelY_vector[i] = np.array(sobelY.flatten())
    i=i+1
feature_vector = np.empty((limit*2,0))
feature_vector=np.append(feature_vector,mean_vector,axis=1)
feature_vector=np.append(feature_vector,var_vector,axis=1)
feature_vector=np.append(feature_vector,std_vector,axis=1)
feature_vector=np.append(feature_vector,skew_vector,axis=1)
feature_vector=np.append(feature_vector,kurto_vector,axis=1)
feature_vector=np.append(feature_vector,entropy_vector,axis=1)
feature_vector=np.append(feature_vector,canny_vector,axis=1)
feature_vector=np.append(feature_vector,sobelX_vector,axis=1)
#feature_vector=np.append(feature_vector,sobelY_vector,axis=1)
#print(feature_vector[0])
#feature_vector=np.append(feature_vector,,axis=1)
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(feature_vector,labels,test_size=0.2,shuffle=True)#80% training

print(xtrain.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

RF=RandomForestClassifier(max_depth=10,n_estimators=150)
RF.fit(xtrain,ytrain)

ypredict = RF.predict(xtest)

print(RF.score(xtest,ytest))
cm = confusion_matrix(ytest, ypredict)

plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True)
plt.title('Confusion Matrix', fontsize = 20)
plt.show()

print(classification_report(ytest,ypredict))
from sklearn import svm
clf = svm.SVC()
clf.fit(xtrain, ytrain)
print(clf.score(xtest,ytest))

ypredict = clf.predict(xtest)

cm = confusion_matrix(ytest, ypredict)

plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True)
plt.title('Confusion Matrix', fontsize = 20)
plt.show()

print(classification_report(ytest,ypredict))
#Bagging classifier
'''from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

clf2 = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100)
clf2.fit(xtrain, ytrain)
print(clf2.score(xtest,ytest))

ypredict = clf2.predict(xtest)

cm = confusion_matrix(ytest, ypredict)

plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True)
plt.title('Confusion Matrix', fontsize = 20)
plt.show()'''
#Takes really long time
from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=[('lr', RF), ('rf',clf)])
vc.fit(xtrain, ytrain)

print(vc.score(xtest,ytest))

ypredict = vc.predict(xtest)

cm = confusion_matrix(ytest, ypredict)

plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True)
plt.title('Confusion Matrix', fontsize = 20)
plt.show()
#KNN Bad results
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain, ytrain)
print(knn.score(xtest,ytest))

ypredict = knn.predict(xtest)

cm = confusion_matrix(ytest, ypredict)

plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True)
plt.title('Confusion Matrix', fontsize = 20)
plt.show()

print(classification_report(ytest,ypredict))
'''import pickle
from joblib import dump, load
dump(RF, 'RF.joblib') 
RF = load('RF.joblib') '''