# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# # import the necessary packages

# import numpy as np

# import cv2
# # load our serialized black and white colorizer model and cluster

# # center points from disk

# print("[INFO] loading model...")

# net = cv2.dnn.readNetFromCaffe("colorization_deploy_v2.prototxt", "colorization_release_v2.caffemodel")

# pts = np.load("pts_in_hull.npy")
# # add the cluster centers as 1x1 convolutions to the model

# class8 = net.getLayerId("class8_ab")

# conv8 = net.getLayerId("conv8_313_rh")

# pts = pts.transpose().reshape(2, 313, 1, 1)

# net.getLayer(class8).blobs = [pts.astype("float32")]

# net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# # load the input image from disk, scale the pixel intensities to the

# # range [0, 1], and then convert the image from the BGR to Lab color

# # space

# image = cv2.imread("output.jpg")

# scaled = image.astype("float32") / 255.0

# lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
# # resize the Lab image to 224x224 (the dimensions the colorization

# # network accepts), split channels, extract the 'L' channel, and then

# # perform mean centering

# resized = cv2.resize(lab, (224, 224))

# L = cv2.split(resized)[0]

# L -= 50



# # pass the L channel through the network which will *predict* the 'a'

# # and 'b' channel values

# 'print("[INFO] colorizing image...")'

# net.setInput(cv2.dnn.blobFromImage(L))

# ab = net.forward()[0, :, :, :].transpose((1, 2, 0))



# # resize the predicted 'ab' volume to the same dimensions as our

# # input image

# ab = cv2.resize(ab, (image.shape[1], image.shape[0]))



# # grab the 'L' channel from the *original* input image (not the

# # resized one) and concatenate the original 'L' channel with the

# # predicted 'ab' channels

# L = cv2.split(lab)[0]

# colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)



# # convert the output image from the Lab color space to RGB, then

# # clip any values that fall outside the range [0, 1]

# colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

# colorized = np.clip(colorized, 0, 1)



# # the current colorized image is represented as a floating point

# # data type in the range [0, 1] -- let's convert to an unsigned

# # 8-bit integer representation in the range [0, 255]

# colorized = (255 * colorized).astype("uint8")



# # show the original and output colorized images

# cv2.imshow("Original", image)

# cv2.imshow("Colorized", colorized)

# cv2.waitKey(0)

# cv2.destroyAllWindows()