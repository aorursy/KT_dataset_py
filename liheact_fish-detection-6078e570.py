import os 

from scipy import ndimage

from subprocess import check_output



import cv2

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline
img_rows, img_cols= 720, 1280

im_array = cv2.imread('../input/train/LAG/img_02236.jpg',0)

template = np.zeros([ img_rows, img_cols], dtype='uint8') # initialisation of the template

template[:, :] = im_array[:,:] # I try multiple times to find the correct rectangle. 

#template /= 255.

max_d = 5

blur_d = 10

min_d = 1



tg_d = min_d

tg_wt = (2*tg_d+1)*(2*tg_d+1)



for i in range(tg_d, img_rows - tg_d - 1):

    for j in range(tg_d, img_cols - tg_d -1):

        #print(np.max(im_array[(i-d):(i+d), (j-d):(j+d)]))

        template[i, j] = np.sum(im_array[(i-tg_d):(i+tg_d), (j-tg_d):(j+tg_d)])/tg_wt

        #template[i, j] = np.max(im_array[(i-max_d):(i+max_d), (j-max_d):(j+max_d)]) - np.max(im_array[(i-min_d):(i+min_d), (j-min_d):(j+min_d)])



for i in range(max_d, img_rows - max_d - 1):

    for j in range(max_d, img_cols - max_d -1):

        #print(np.max(im_array[(i-d):(i+d), (j-d):(j+d)]))

        #template[i, j] = np.sum(im_array[(i-tg_d):(i+tg_d), (j-tg_d):(j+tg_d)])/tg_wt

        maxc = np.max(template[(i-max_d):(i+max_d), (j-max_d):(j+max_d)])

        minc = np.min(template[(i-min_d):(i+min_d), (j-min_d):(j+min_d)])

        template[i, j] =  (maxc-minc)*255.0/maxc

        #template[i, j] = np.max(template[(i-max_d):(i+max_d), (j-max_d):(j+max_d)]) - template[i, j]

plt.subplots(figsize=(100, 70))

plt.subplot(121),plt.imshow(template, cmap='gray') 

plt.subplot(122), plt.imshow(im_array, cmap='gray')


file_name = '../input/train/LAG/img_01512.jpg' # img_00176,img_02758, img_01512

img = cv2.imread(file_name,0) 

img2 = img

w, h = template.shape[::-1]



# All the 6 methods for comparison in a list

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',

            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']



for meth in methods:

     img = img2

     method = eval(meth)

 

     # Apply template Matching

     res = cv2.matchTemplate(img,template,method)

     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

 

     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:

         top_left = min_loc

     else:

         top_left = max_loc

     bottom_right = (top_left[0] + w, top_left[1] + h)

 

     cv2.rectangle(img,top_left, bottom_right, 255, 2)

     fig, ax = plt.subplots(figsize=(12, 7))

     plt.subplot(121),plt.imshow(res,cmap = 'gray')

     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

     plt.subplot(122),plt.imshow(img,cmap = 'gray') #,aspect='auto'

     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

     plt.suptitle(meth)

 

     plt.show()


method = eval('cv2.TM_CCOEFF')

indexes=[1,30,40,5]



train_path = "../input/train/"

sub_folders = check_output(["ls", train_path]).decode("utf8").strip().split('\n')

for sub_folder in sub_folders:

    file_names = check_output(["ls", train_path+sub_folder]).decode("utf8").strip().split('\n')

    k=0

    _, ax = plt.subplots(2,2,figsize=(10, 7))

    for file_name in [file_names[x] for x in indexes]: # I take only 4 images of each group. 

        img = cv2.imread(train_path+sub_folder+"/"+file_name,0)

        img2 = img

        w, h = template.shape[::-1]

        # Apply template Matching

        res = cv2.matchTemplate(img,template,method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

 

        cv2.rectangle(img,top_left, bottom_right, 255, 2)

        if k==0 : 

            ax[0,0].imshow(img,cmap = 'gray')

            plt.xticks([]), plt.yticks([])

        if k==1 : 

            ax[0,1].imshow(img,cmap = 'gray')

            plt.xticks([]), plt.yticks([])

        if k==2 : 

            ax[1,0].imshow(img,cmap = 'gray')

            plt.xticks([]), plt.yticks([])

        if k==3 : 

            ax[1,1].imshow(img,cmap = 'gray')

            plt.xticks([]), plt.yticks([])

        k=k+1

    plt.suptitle(sub_folder)

    plt.show()