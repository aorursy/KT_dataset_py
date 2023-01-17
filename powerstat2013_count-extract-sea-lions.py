import pandas as pd

import numpy as np

import glob

import os

import cv2

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
submission.shape
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
index = 1

print(index)

sl_counts = train_data.iloc[index]

print(sl_counts)

print('[Ground Truth] Sea Lion Count: {}'.format(sum(sl_counts[1:])))



img = cv2.cvtColor(cv2.imread(train_imgs[index]), cv2.COLOR_BGR2RGB)

img_dot = cv2.cvtColor(cv2.imread(train_dot_imgs[index]), cv2.COLOR_BGR2RGB)





# Now create a mask of missing area in dotted image and create its inverse mask also

img_dot_gray = cv2.cvtColor(img_dot,cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img_dot_gray, 10, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)



img_dot_missing_area = cv2.bitwise_and(img,img,mask = mask_inv)

fnl_dot = cv2.add(img_dot,img_dot_missing_area)



###################################################



"""

diff = cv2.absdiff(img_dot, img)

diff[img_dot==0] = 255

diff = np.max(diff, axis=-1)

f, ax = plt.subplots(1,1,figsize=(16,30))

ax.imshow(diff, 'gray')

"""





diff = cv2.absdiff(fnl_dot, img)

gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)





ret,th1 = cv2.threshold(gray,10,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)



# All contours

cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

print("[Initial Counted] Sea Lions Found: {}".format(len(cnts)))



# Remove contours where radius is very very small

pruned_contours = []

min_radius_threshold = 1.5

max_radius_threshold = 5.0

for (i, c) in enumerate(cnts):

    ((x, y), r) = cv2.minEnclosingCircle(c)

    if ((r>=min_radius_threshold) & (r<=max_radius_threshold)):

        pruned_contours.append(c)

print("[Final Counted] Sea Lions Found: {}".format(len(pruned_contours)))



#for (i, c) in enumerate(pruned_contours):

#    ((x, y), _) = cv2.minEnclosingCircle(c)

#    cv2.putText(diff, "{}".format(i + 1), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#    cv2.drawContours(diff, [c], -1, (0, 255, 0), 2)







#f, ax = plt.subplots(1,1,figsize=(16,30))

#(ax1,ax2) = ax.flatten()

#ax.imshow(diff)

#ax.imshow(gray,'gray')