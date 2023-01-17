%matplotlib inline



import numpy as np

import cv2

from skimage.measure import compare_ssim

import os

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
def bboxes_car(path, img_id, debug=False):

    # You need to input the path (train or test) and the car_id (img_id)

    bboxes = []

    # Let's iterate over all the angles

    for num in range(1, 17):

        # Here we read images i and i+1. 

        # If i==16, we will read the first image

        # To speed up the things, we can scale the images 5 times

        fname1 = os.path.join(path, img_id+ '_{:0>2}.jpg'.format(num))

        fname2 = os.path.join(path, img_id+ '_{:0>2}.jpg'.format((num) % 16+1))

        img_1_orig = cv2.imread(fname1)

        h, w = img_1_orig.shape[0],img_1_orig.shape[1],

        img_1_scaled = cv2.resize(img_1_orig, (w//5, h//5))



        img_2_orig = cv2.imread(fname2)

        h, w = img_2_orig.shape[0],img_2_orig.shape[1],

        img_2_scaled = cv2.resize(img_2_orig, (w//5, h//5))



        if debug:

            plt.figure()

            plt.subplot(121)

            plt.title('Current image [{}]'.format(num))

            plt.imshow(img_1_scaled)

            plt.subplot(122)

            plt.title('Next image [{}]'.format((num) % 16+1))

            plt.imshow(img_2_scaled)

            plt.show()

        

        # As the images differ from each other just a by a small angle of rotation,

        # We can find their difference and draw a boundign box around the obtained image

        img1 = cv2.cvtColor(img_1_scaled, cv2.COLOR_BGR2GRAY)

        img2 = cv2.cvtColor(img_2_scaled, cv2.COLOR_BGR2GRAY)



        # Instead of plain difference, we look for structural similarity

        score, dimg = compare_ssim(img1, img2, full=True)

        dimg = (dimg * 255).astype("uint8")





        thresh = cv2.threshold(dimg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        if debug:

            plt.figure()

            plt.title('Difference image')

            plt.imshow(dimg>thresh)

            plt.show()        

        

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        

        ROIS = []

        for c in cnts[1]:

            (x, y, w, h) = cv2.boundingRect(c)

            # We dont want to use too small bounding boxes

            if w*h > img1.shape[0]*img1.shape[1]//9:

                ROIS.append([x, y, x+w, y+h])



        ROIS = np.array(ROIS)



        # Now we will draw a boundig box 

        # around all the bounding boxes (there are outliers)

        x1 = ROIS[:,0].min()

        y1 = ROIS[:,1].min()



        x2 = ROIS[:,2].max()

        y2 = ROIS[:,3].max()



        if debug:

            plt.figure()

            plt.imshow(img_1_orig)

            rect = Rectangle((x1*5, y1*5), (x2-x1)*5, (y2-y1)*5, fill=False, color='red')

            plt.axes().add_patch(rect)

            plt.show()      

        bboxes.append([fname1, x1*5, y1*5, x2*5, y2*5])

    return bboxes

car_id = '0cdf5b5d0ce1'

path = '../input/train/'



bboxes_car = bboxes_car(path, car_id, debug=True)