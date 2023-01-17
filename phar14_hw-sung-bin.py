! kaggle competitions download -c 2019-fall-pr-project





import zipfile

import os



os.mkdir('/content/input2')



zip_ref = zipfile.ZipFile("/content/2019-fall-pr-project.zip", 'r')

zip_ref.extractall("/content/input2")

zip_ref.close()


from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from imutils import paths

import numpy as np

import imutils 

import cv2 

import os

import argparse







def image_to_feature_vector(image, size=(32, 32)):

    return cv2.resize(image, size).flatten()



dataset_train = "/content/input2/train/"



print("[INFO] describing images...")

imagePaths = list(paths.list_images(dataset_train))



print(len(imagePaths))

# initialize the raw pixel intensities matrix, the features matrix,

# and labels list

rawImages = []

labels = []

for (i, imagePath) in enumerate(imagePaths):

	# load the image and extract the class label (assuming that our

	# path as the format: /path/to/dataset/{class}.{image_num}.jpg



    image = cv2.imread(imagePath)

    label = imagePath.split(os.path.sep)[-1].split(".")[0]

 

# extract raw pixel intensity "features", followed by a color

	# histogram to characterize the color distribution of the pixels

	# in the image



    pixels = image_to_feature_vector(image)

    

 

# update the raw images, features, and labels matricies,

	# respectively

    rawImages.append(pixels)

    labels.append(label)

 

# show an update every 1,000 images

    if i > 0 and i % 1000 == 0:

        print("[INFO] processed {}/{}".format(i, len(imagePaths)))


rawImages = np.array(rawImages)

labels = np.array(labels)



print("[INFO] pixels matrix: {:.2f}MB".format(

rawImages.nbytes / (1024 * 1000.0)))

bUseCompleteDataset = False

if bUseCompleteDataset:

    (trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)

else:

    rawImages_subset = rawImages[:2000]

    labels_subset = labels[:2000]

    (trainRI, testRI, trainRL, testRL) = train_test_split(rawImages_subset, labels_subset, test_size=0.25, random_state=42)











# train and evaluate a k-NN classifer on the raw pixel intensities



print("[INFO] evaluating raw pixel accuracy...")

neighbors = [1, 3, 5, 7, 13]



for k in neighbors:

    model = KNeighborsClassifier(n_neighbors= 5)

    a=model.fit(trainRI, trainRL)

    acc = model.score(testRI, testRL)

    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

dataset_test = "/content/input2/test1/"



imagePaths_test=list(paths.list_images(dataset_test))



print(len(imagePaths_test))

rawImages_test=[]

labels_test=[]



for(i,imagePath_test) in enumerate(imagePaths_test):

    image=cv2.imread(imagePath_test)

    label=imagePath_test.split(os.path.sep)[-1].split(".")[0]



    pixels=image_to_feature_vector(image)



    rawImages_test.append(pixels)

    labels_test.append(label)





    if i>0 and i%1000==0:

      print("[INFO] processed {}/{}".format(i,len(imagePaths_test)))



model = KNeighborsClassifier(n_neighbors= 5)

a=model.fit(trainRI, trainRL)

result=a.predict(rawImages_test)
#numpy 를 Pandas 이용하여 결과 파일로 저장

import pandas as pd





result=np.reshape(result,(-1,1))



print(result.shape)

df = pd.DataFrame(result, columns=["label"])

df.index.name = 'id'

df.index += 1

df = df.replace('dog',1)

df = df.replace('cat',0)

df = df.rename({'1':'id','0':'label'})

df.to_csv('sungbin-v2.csv',index=True, header=True)