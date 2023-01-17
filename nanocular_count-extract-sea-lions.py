import pandas as pd

import numpy as np

import glob

import os

import cv2

import sys

from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline



train_data = pd.read_csv('../input/Train/train.csv')

train_imgs = sorted(glob.glob('../input/Train/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))

train_dot_imgs = sorted(glob.glob('../input/TrainDotted/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))



submission = pd.read_csv('../input/sample_submission.csv')





print(train_data.shape)

print('Number of Train Images: {:d}'.format(len(train_imgs)))

print('Number of Dotted-Train Images: {:d}'.format(len(train_dot_imgs)))







print(train_data.head(6))



#test_imgs = glob.glob('../input/Test/*.jpg')

#print('Number of Test Images: {:d}'.format(len(test_imgs)))

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
# Count of each type

hist = train_data.sum(axis=0)

print(hist)





sea_lions_types = hist[1:]

f, ax1 = plt.subplots(1,1,figsize=(5,5))

sea_lions_types.plot(kind='bar', title='Count of Sea Lion Types (Train)', ax=ax1)

plt.show()
index = 5

sl_counts = train_data.iloc[index]

print(sl_counts)



plt.figure()

sl_counts.plot(kind='bar', title='Count of Sea Lion Types')

plt.show()



print(train_imgs[index])

img = cv2.cvtColor(cv2.imread(train_imgs[index]), cv2.COLOR_BGR2RGB)

img_dot = cv2.cvtColor(cv2.imread(train_dot_imgs[index]), cv2.COLOR_BGR2RGB)



crop_img = img[200:2000, 2600:3500]

crop_img_dot = img_dot[200:2000, 2600:3500]



f, ax = plt.subplots(1,2,figsize=(16,8))

(ax1, ax2) = ax.flatten()



ax1.imshow(img)

ax2.imshow(img_dot)



plt.show()
crop_img = img[1350:1900, 3000:3400]

crop_img_dot = img_dot[1350:1900, 3000:3400]



f, ax = plt.subplots(1,2,figsize=(16,8))

(ax1, ax2) = ax.flatten()



ax1.imshow(crop_img)

ax2.imshow(crop_img_dot)



plt.show()
index = 5



image = cv2.cvtColor(cv2.imread(train_imgs[index]), cv2.COLOR_BGR2RGB)

image_dot = cv2.cvtColor(cv2.imread(train_dot_imgs[index]), cv2.COLOR_BGR2RGB)



img = image[1350:1900, 3000:3400]

img_dot = image_dot[1350:1900, 3000:3400]



#img_c = np.copy(img)



diff = cv2.absdiff(img_dot, img)

gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

ret,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#plt.figure(figsize=(16,8))

#plt.imshow(th1, 'gray')



cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

print("Sea Lions Found: {}".format(len(cnts)))



for (i, c) in enumerate(cnts):

	((x, y), _) = cv2.minEnclosingCircle(c)

	cv2.putText(diff, "{}".format(i + 1), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

	cv2.drawContours(diff, [c], -1, (0, 255, 0), 2)



#plt.figure(figsize=(16,8))

#plt.imshow(diff)



f, ax = plt.subplots(3,1,figsize=(18,35))

(ax1, ax2, ax3) = ax.flatten()

ax1.imshow(img_dot)

ax2.imshow(th1, 'gray')

ax3.imshow(diff)

#plt.show()
def get_sl_classification(color):

	# return classification based on least euclidean distance of color

	in_color = np.array(color)

	sealion_colors = {'adult male': np.array((255, 0, 0)), 'subadult male': np.array((255, 0, 255)), 'pup': np.array((0, 175, 0)), 'juvenile': np.array((0,0,190)), 'adult female': np.array((80,40,0))}

	closest_match = sys.maxsize

	sl_type = None

	for key, value in sealion_colors.items():

		distance = np.linalg.norm(in_color - value)

		if distance < closest_match:

			closest_match = distance

			sl_type = key

	return sl_type



index = 5



sl_counts = train_data.iloc[index]

print('[Ground Truth] Sea Lion Count: {}'.format(sum(sl_counts[1:])))



image = cv2.cvtColor(cv2.imread(train_imgs[index]), cv2.COLOR_BGR2RGB)

image_dot = cv2.cvtColor(cv2.imread(train_dot_imgs[index]), cv2.COLOR_BGR2RGB)

image_dot_col = Image.open(train_dot_imgs[index])

#    (w, h) = image_dot_col.size

box = (2800, 0, 3950, 3230)

image_dot_col = image_dot_col.crop(box)

px = image_dot_col.load()



img = image[:3230, 2800:3950]

img_dot = image_dot[0:3230, 2800:3950]



#img = image[:, :]

#img_dot = image_dot[:, :]



diff = cv2.absdiff(img_dot, img)

gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

ret,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)



cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

print("[Counted] Sea Lions Found: {}".format(len(cnts)))

sealion_counts = dict()

for (i, c) in enumerate(cnts):

	((x, y), _) = cv2.minEnclosingCircle(c)

	cv2.putText(diff, "{}".format(i + 1), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

	cv2.drawContours(diff, [c], -1, (0, 255, 0), 2)

	color = px[x, y]

	x = int(x)

	y = int(y)



	class_type = get_sl_classification(color)

	if class_type in sealion_counts:

		sealion_counts[class_type] += 1

	else:

		sealion_counts[class_type] = 1



print(sealion_counts)





f, ax = plt.subplots(1,2,figsize=(16,15))

(ax1, ax2) = ax.flatten()

ax1.imshow(img_dot)

ax2.imshow(diff)



#plt.figure(figsize=(18,25))

#plt.imshow(diff)
        