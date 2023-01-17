import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import seaborn as sns

import cv2

import os

import glob

print(os.listdir("../input"))
images_s01 = [cv2.imread(file) for file in glob.glob("../input/facess/subject*.png")]

images_s06 = [cv2.imread(file) for file in glob.glob("../input/facesss/subject6*.png")]

images_s07 = [cv2.imread(file) for file in glob.glob("../input/facesss/subject7*.png")]

images_s08 = [cv2.imread(file) for file in glob.glob("../input/facesss/subject8*.png")]

images_s09 = [cv2.imread(file) for file in glob.glob("../input/facesss/subject9*.png")]

print('no of s1 images: '+str(len(images_s07)))
directory = [images_s01,images_s06,images_s07,images_s08,images_s09]

imgs = []

for i in directory: 

    for img in i:

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #rbg to gray                    

        img = cv2.resize(img,(200,200)) # resize image to 20x20

        imgs.append(img.flatten()) #flatten array
label = [] # making labels for faces

for i in range(len(directory)):

    for j in range(len(images_s01)):

        label.append(i)
X_train, X_test, Y_train, Y_test = tts(imgs, label, test_size = 0.3, random_state = 3)

print('Length of train dataset ' + str(len(X_train)) + ', Length of test dataset ' +str(len(X_test)))
def PCA(X_train, k):

    mean_normalized = X_train - np.mean(X_train)

    covariance_t = np.cov(mean_normalized) # covariance_t = np.cov(mean_normalized.T) <- this is computationally exapensive

    #covariance_t = covariance_t.T

    eigen_values, eigen_vectors = np.linalg.eig(covariance_t)

    # select best k eigen vectors

    eigen_vectors_pd = pd.DataFrame(data=eigen_vectors)

    sorted_indices = np.argsort(eigen_values)

    sorted_indices_k = sorted_indices[:k]

    eigen_vectors_pd_k = eigen_vectors_pd[sorted_indices_k]

    eigen_vectors_k = eigen_vectors_pd_k.values

    eigen_faces = np.dot(mean_normalized.T,eigen_vectors_k)

    projected_faces = np.dot(mean_normalized, eigen_faces)

    return eigen_faces,projected_faces
h = PCA(X_train, 38) #30 is optimal

print(h[1].shape)
def predict(X, eigen_faces, projected_faces, Y_train):

    mean_normalized = X - np.mean(X_train)

    

    projected_face = np.dot(mean_normalized, eigen_faces)

    all_diff = []

    for face in projected_faces:

        diff = np.linalg.norm(face-projected_face)

        all_diff.append(diff)

    iloc = np.argmin(all_diff)

    predicted = Y_train[iloc]

    return predicted
predicted_values = [predict(x, h[0], h[1], Y_train) for x in X_test]
m = confusion_matrix(Y_test, predicted_values)

ax = sns.heatmap(m)

print("Confusion Matrix",cm)
print("accuracy: "+str(accuracy_score(Y_test, predicted_values)*100))