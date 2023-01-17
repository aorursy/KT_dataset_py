import cv2

import matplotlib.pyplot as plt

import numpy as np

!pip install imutils

import imutils

from imutils import paths

import os

from bs4 import BeautifulSoup # u/ buka xml
# buat directory helmet dan non-helmet

os.mkdir("/kaggle/working/helmet/")

os.mkdir("/kaggle/working/non-helmet/")
# definisi image paths

imagePaths = list(paths.list_images("../input/hardhat-v3/Images"))

annotPaths = "../input/hardhat-v3/Annotations"



helmetPath = "helmet"

nonHelmetPath = "non-helmet"



totalHelmet = 0

totalNonHelmet = 0



for (i, imagePath) in enumerate(imagePaths):

    

    print("[INFO] preprocessing image {}/{} ...".format(i + 1, len(imagePaths)))

    

    

    filename = imagePath.split(os.path.sep)[-1] ## nama image 206.jpg

    

    # hapus ekstensi jpg

    filename = filename[:filename.rfind(".")]

    

    annotPath = os.path.sep.join([annotPaths, "{}.xml".format(filename)])

    

    # buka anotasi

    contents = open(annotPath).read()

    soup = BeautifulSoup(contents, "html.parser")

    

    # inisiasi array penyimpan bounding box

    helmetBoxes = []

    nonHelmetBoxes = []

    

    # extract dimensi image

    w = int(soup.find("width").string)

    h = int(soup.find("height").string)

    

    # loop pada element "object" yg ditemukan didalam xml

    for o in soup.find_all("object"):

        # extract label dan bounding box

        label = o.find("name").string

        

        if label == "hat":

            xMin = int(o.find("xmin").string)

            yMin = int(o.find("ymin").string)

            xMax = int(o.find("xmax").string)

            yMax = int(o.find("ymax").string)



            # pastikan bounding box tidak keluar dari dimensi image

            xMin = max(0, xMin)

            yMin = max(0, yMin)

            xMax = min(w, xMax)

            yMax = min(h, yMax)

            helmetBoxes.append((xMin, yMin, xMax, yMax))

        

        else:

            xMin = int(o.find("xmin").string)

            yMin = int(o.find("ymin").string)

            xMax = int(o.find("xmax").string)

            yMax = int(o.find("ymax").string)



            # pastikan bounding box tidak keluar dari dimensi image

            xMin = max(0, xMin)

            yMin = max(0, yMin)

            xMax = min(w, xMax)

            yMax = min(h, yMax)

            nonHelmetBoxes.append((xMin, yMin, xMax, yMax))

            

    # ambil image dari dataset

    image = cv2.imread(imagePath)

    

    

    for bbox in helmetBoxes:

        (startX, startY, endX, endY) = bbox

        roi = image[startY:endY, startX:endX]

        roi = cv2.resize(roi, (224,224), interpolation=cv2.INTER_CUBIC)

        filename = "{}.png".format("helmet-" + str(totalHelmet))

        outputPath = os.path.sep.join([helmetPath, filename])        

        totalHelmet += 1

        

        cv2.imwrite(outputPath, roi)

    

    for bbox in nonHelmetBoxes:

        (startX, startY, endX, endY) = bbox

        roi = image[startY:endY, startX:endX]

        roi = cv2.resize(roi, (224,224), interpolation=cv2.INTER_CUBIC)

        filename = "{}.png".format("non-helmet-" + str(totalNonHelmet))

        outputPath = os.path.sep.join([nonHelmetPath, filename])

        totalNonHelmet += 1

        

        cv2.imwrite(outputPath, roi)

        

    helmetBoxes = []

    nonHelmetBoxes = []

    

#     if i >= 500:

#         print("[INFO] processing done...")

#         print("[INFO] Helmet dataset :", totalHelmet)

#         print("[INFO] Non Helmet dataset :", totalNonHelmet)

#         break



print("[INFO] processing done...")

print("[INFO] helmet dataset :", totalHelmet)

print("[INFO] non-helmet dataset :", totalNonHelmet)