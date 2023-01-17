import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import cv2

import io

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from matplotlib import cm

import os



listFace = []
archive = os.listdir('../input/faces-data-new/images')
listImages = [name for name in archive if name.endswith('.jpg')]

listImages.sort()

listLabels = [name.replace('.jpg','').split('.')[0] for name in listImages if name.endswith('.jpg')]
print(len(listImages))

print(len(listLabels))
faceCascade = cv2.CascadeClassifier('../input/haarcascadefrontalfaces/haarcascade_frontalface_default.xml')



img = ''



img = cv2.imread('../input/faces-data-new/images/'+listImages[0], cv2.IMREAD_COLOR)



img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)



faces = faceCascade.detectMultiScale(

    img,     

    scaleFactor=1.2,

    minNeighbors=5,     

    minSize=(20, 20)

)



for x,y,w,h in faces:

    img = img[y:y+h,x:x+w]

    



plt.subplot(1, 2, 1)

plt.imshow(img, cmap = plt.cm.gray)



hist, bins = np.histogram(img.flatten(), 256, [0, 256])



cdf = hist.cumsum()

cdf_normalized = cdf * hist.max() / cdf.max()



plt.subplot(1, 2, 2)

plt.plot(cdf_normalized, color = 'b')

plt.hist(img.flatten(), 256, [0,256], color='r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()
for index in np.arange(len(listImages)):

    img = ''



    img = cv2.imread('../input/faces-data-new/images/'+listImages[index], cv2.IMREAD_COLOR)



    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            

    faces = faceCascade.detectMultiScale(

        img,     

        scaleFactor=1.2,

        minNeighbors=5,     

        minSize=(20, 20)

    )



    for x,y,w,h in faces:

        img = img[y:y+h,x:x+w]

        

    if img.shape[1] > 0 and img.shape[0] > 0:

        img = cv2.resize(img, (80,100), interpolation = cv2.INTER_AREA)



        hist, bins = np.histogram(img.flatten(), 256, [0, 256])



        cdf = hist.cumsum()

        cdf_normalized = cdf * hist.max() / cdf.max(),

        

        cdf_m = np.ma.masked_equal(cdf, 0)

        cdf_m = (cdf_m - cdf_m.min())*255 / (cdf_m.max()-cdf_m.min())

        cdf = np.ma.filled(cdf_m, 0).astype('uint8')



        img = cdf[img]

        #_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        listFace.append(img.reshape(80*100))

        

    else:

        #listLabels.iloc[index, 0] = float("NaN")

        listLabels[index] = float("NaN")
listFace = np.array(listFace)

listFace.shape
img = listFace[0].reshape(100,80)

plt.subplot(1, 2, 1)

plt.imshow(img, cmap = plt.cm.gray)



hist, bins = np.histogram(img.flatten(), 256, [0, 256])



cdf = hist.cumsum()

cdf_normalized = cdf * hist.max() / cdf.max()



plt.subplot(1, 2, 2)

plt.plot(cdf_normalized, color = 'b')

plt.hist(img.flatten(), 256, [0,256], color='r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()
listLabels = pd.DataFrame(listLabels)
listLabels.dropna(inplace = True)

listLabels.reset_index(inplace = True, drop=True)
dic_class = {value: index for index, value in enumerate(listLabels[0].unique())}



listLabels[0] = listLabels[0].apply(lambda value: dic_class[value])
x = pd.DataFrame(listFace)



X_train, X_test, y_train, y_test = train_test_split(x.values, listLabels.values, test_size=0.3)
imageMean = np.array(list(map(lambda index: X_train[:, index].mean(), np.arange(X_train.shape[1]))))



plt.imshow(imageMean.reshape(100,80), cmap = plt.cm.gray, interpolation='sinc')
dataMean = X_train - imageMean
pca = PCA(n_components=40, copy=False)

pc_df = pca.fit_transform(dataMean)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_) * 100
plt.imshow(X_train[0].reshape(100,80),

              cmap = plt.cm.gray, interpolation='nearest',

              clim=(0, 255))

plt.xlabel(str(100*80) + ' components', fontsize = 14)

plt.title('Original Image', fontsize = 20)
plt.figure(figsize=(20,10))

length_pca = pca.n_components_



for index, value in enumerate(range(0, length_pca, 5)):

    data_original = np.dot(pc_df[:,:value], pca.components_[:value, :]) + imageMean

    plt.subplot(1, (length_pca/5)+1, (value/5)+1)

    plt.imshow(data_original[0].reshape(100, 80),

                  cmap = plt.cm.gray)

    plt.xlabel(str(value) + ' components', fontsize = 10)

    plt.title('{:.2f}%'.format(sum(pca.explained_variance_ratio_[:value]) * 100), fontsize = 9.5)



data_original = np.dot(pc_df[:,:40], pca.components_[:40, :]) + imageMean

plt.subplot(1, 9, 9)

plt.imshow(data_original[0].reshape(100, 80),

              cmap = plt.cm.gray)

plt.xlabel(str(40) + ' components', fontsize = 10)

plt.title('{:.2f}% Explained Variance'.format(sum(pca.explained_variance_ratio_[:40]) * 100), fontsize = 9.5)
plt.figure(figsize=(20,10))

length_pca = pca.n_components_



for index, value in enumerate(range(0, length_pca, 5)):

    data_original = np.dot(pc_df[:,:value], pca.components_[:value, :]) + imageMean

    plt.subplot(1, (length_pca/5)+1, (value/5)+1)

    plt.imshow(data_original[0].reshape(100, 80),

                  cmap = cm.coolwarm, interpolation='sinc')

    plt.xlabel(str(value) + ' components', fontsize = 10)

    plt.title('{:.2f} Explained Variance'.format(sum(pca.explained_variance_ratio_[:value]) * 100), fontsize = 9.5)



data_original = np.dot(pc_df[:,:40], pca.components_[:40, :]) + imageMean

plt.subplot(1, 9, 9)

plt.imshow(data_original[0].reshape(100, 80),

              cmap = cm.coolwarm)

plt.xlabel(str(40) + ' components', fontsize = 10)

plt.title('{:.2f} Explained Variance'.format(sum(pca.explained_variance_ratio_[:40]) * 100), fontsize = 9.5)
lda = LinearDiscriminantAnalysis()
lda = lda.fit(pc_df, y_train)

lda_df = lda.transform(pc_df)
lda.explained_variance_ratio_
lda.explained_variance_ratio_.shape
lda_df.shape
test1 = lda_df[:, 0]

for i in np.arange(1, lda_df.shape[1]/2):

    test1 = test1 + lda_df[:, int(i)]



test2 = lda_df[:, 15]

for i in np.arange(lda_df.shape[1]/2 + 2, lda_df.shape[1]):

    test2 = test2 + lda_df[:, int(i)]
plt.scatter(test1, test2, c=y_train, s=30, cmap='Set1')
pc_test = pca.transform(X_test)
lda.score(pc_test, y_test)