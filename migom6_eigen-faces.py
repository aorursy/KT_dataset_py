import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import seaborn as sns

import cv2

import os

import glob

print(os.listdir("../input/faces/faces"))
images_s01 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s01/*.png")]

images_s02 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s02/*.png")]

images_s03 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s03/*.png")]

images_s04 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s04/*.png")]

images_s05 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s05/*.png")]

images_s06 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s05/*.png")]

images_s07 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s05/*.png")]

images_s08 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s05/*.png")]

images_s09 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s05/*.png")]

images_s10 = [cv2.imread(file) for file in glob.glob("../input/faces/faces/s05/*.png")]
print('no of s1 images: '+str(len(images_s01)))

print('no of s2 images: '+str(len(images_s02)))

print('no of s3 images: '+str(len(images_s03)))

print('no of s4 images: '+str(len(images_s04)))

# print('no of s5 images: '+str(len(images_s05)))

# print('no of s6 images: '+str(len(images_s06)))

# print('no of s7 images: '+str(len(images_s07)))

# print('no of s8 images: '+str(len(images_s08)))

# print('no of s9 images: '+str(len(images_s09)))

# print('no of s10 images: '+str(len(images_s10)))



all_images_with_directory = [images_s01,images_s02,images_s03,images_s04]

# all_images_with_directory = [images_s01,images_s02,images_s03,images_s04,images_s05,images_s06,images_s07,images_s08,images_s09,images_s10]
all_images = []

for i in all_images_with_directory:

    for img in i:

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #rbg to gray                    

        img = cv2.resize(img,(200,200)) # resize image to 20x20

        all_images.append(img.flatten()) #flatten array
labels = [] # making labels for faces

for i in range(len(all_images_with_directory)):

    for j in range(len(images_s01)):

        labels.append(i)
X_train, X_test, Y_train, Y_test = tts(all_images, labels, test_size = 0.3, random_state = 5)

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

h = PCA(X_train, 30) #30 is optimal

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
cm = confusion_matrix(Y_test, predicted_values)

print("Confusion Matrix",cm)

ax = sns.heatmap(cm)
print("accuracy: "+str(accuracy_score(Y_test, predicted_values)*100))
# from sklearn.decomposition import PCA

# pca = PCA(.99)

# k = pca.fit(X_train)
# len(pca.transform(X_train)[0])