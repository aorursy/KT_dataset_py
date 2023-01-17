import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sys

import matplotlib.pyplot as plt

%matplotlib inline

from zipfile import ZipFile

from PIL import Image

from matplotlib.pyplot import imshow

import cv2



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sample_df = pd.read_csv("../input/sample_submission.csv")

# print(train_df.shape)

# print(test_df.shape)





plate_names = ["Plate1", "Plate2", "Plate3", "Plate4"]

test_experiment_names = [name for name in os.listdir("../input/test")]

train_experiment_names = [name for name in os.listdir("../input/train")]
train_df.head()
ax = train_df["sirna"].plot.hist(bins=max(train_df["sirna"]))
test_df.head()
sample_df.head()
print(os.listdir("../input"))

print(os.listdir("../input/train"))
file = "../input/test/HEPG2-11/Plate1/G07_s2_w1.png"

img = Image.open(file)

imshow(np.asarray(img))
# This will show 5 images for each plate from test dataset

plate_n = 0

for exp in test_experiment_names:

    for plate in plate_names:

        path = "../input/test/" + exp + "/" + plate + "/"

        print(exp + "/" + plate)

        plt.figure(figsize=(18, 16))

        for image_path in os.listdir(path):

            file = path + image_path

            img = Image.open(file)

            plt.subplot(1,5,plate_n+1), plt.imshow(img)

            

            plate_n = plate_n + 1

            if plate_n == 5:

                break

        plt.show()

        plate_n = 0
# This will show 5 images for each plate from train dataset

plate_n = 0

for exp in train_experiment_names:

    for plate in plate_names:

        path = "../input/train/" + exp + "/" + plate + "/"

        print(exp + "/" + plate)

        plt.figure(figsize=(18, 16))

        for image_path in os.listdir(path):

            file = path + image_path

            img = Image.open(file)

            plt.subplot(1,5,plate_n+1), plt.imshow(img)

            

            plate_n = plate_n + 1

            if plate_n == 5:

                break

        plt.show()

        plate_n = 0
plate_n = 0

# Check all sirnas groupm

for label in range(max(train_df["sirna"])):

    label1 = train_df[train_df["sirna"] == label]

#     print("Sirna == ", label)
label2 = train_df[train_df["sirna"] == 1]



for exp in train_experiment_names:

    if exp not in list(label2["experiment"]):

        continue

    

    # Find index of exp found

    idx = label2.index[label2['experiment'] == exp].tolist()[0]

    

    # Find which plate and well

    plate = "Plate" + str(label2["plate"][idx])

    well = str(label2["well"][idx])

    

    path = "../input/train/" + exp + "/" + plate + "/"

    print(exp + "/" + plate, well)

    plt.figure(figsize=(18, 16))

    

    for image_path in os.listdir(path):

        if image_path.split(well)[0] == "":

            file = path + image_path

            img = Image.open(file)

            plt.subplot(1,5,plate_n+1), plt.imshow(np.asarray(img))

            

            plate_n = plate_n + 1

            if plate_n == 5:

                break

                

    plt.show()

    plate_n = 0
img = cv2.imread(file,0)



hist,bins = np.histogram(img.flatten(),256,[0,256])



cdf = hist.cumsum()

cdf_normalized = cdf * hist.max()/ cdf.max()



plt.plot(cdf_normalized, color = 'b')

plt.hist(img.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()

img = cv2.imread(file,0)

equ = cv2.equalizeHist(img)

res = np.hstack((img, equ))

plt.imshow(res)

plt.show()

hist,bins = np.histogram(equ.flatten(),256,[0,256])



cdf = hist.cumsum()

cdf_normalized = cdf * hist.max()/ cdf.max()



plt.plot(cdf_normalized, color = 'b')

plt.hist(img.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()
img.shape
img = cv2.imread(file)

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))



cl1 = clahe.apply(img_gray)

res = np.hstack((img_gray, cl1))

plt.imshow(cl1)

plt.show()
img = cv2.imread(file)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,thresh_trunc = cv2.threshold(gray_image,50,255,cv2.THRESH_TRUNC)

ret,thresh_tozero_inv = cv2.threshold(gray_image,50,255,cv2.THRESH_TOZERO_INV)



#DISPLAYING THE DIFFERENT THRESHOLDING STYLES

names = ['Original Image','THRESH_TRUNC','THRESH_TOZERO_INV']

images = gray_image,thresh_trunc,thresh_tozero_inv



plt.figure(figsize=(18, 16))

for i in range(3):

    plt.subplot(1,3,i+1),plt.imshow(images[i],'gray')

    plt.title(names[i])

    plt.xticks([]),plt.yticks([])

    

plt.show()
ret,thresh_global = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)



thresh_mean = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

thresh_gaussian = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)



names = ['Original Image','Global Thresholding','Adaptive Mean Threshold','Adaptive Gaussian Thresholding']

images = [gray_image,thresh_global,thresh_mean,thresh_gaussian]



plt.figure(figsize=(18, 16))

for i in range(4):

    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')

    plt.title(names[i])

    plt.xticks([]),plt.yticks([])

    

plt.show()
#using the averaging kernel for image smoothening 

averaging_kernel = np.ones((3,3),np.float32)/3

filtered_image = cv2.filter2D(img, -1, averaging_kernel)

plt.imshow(filtered_image)