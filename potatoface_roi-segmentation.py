import numpy as np

import cv2

import matplotlib.pyplot as plt
def focus(im_prefix, w=80, h=80, h_border=40, v_border=40):

    mask = cv2.imread("../input/train/" + im_prefix + "_mask.tif", -1)

    assert mask.max() > 0, "Mask not present"



    im = cv2.imread("../input/train/" + im_prefix + ".tif", -1)



    im2,contours,hierarchy = cv2.findContours(mask.copy(), 1, 2)

    cnt = contours[0]

    M = cv2.moments(cnt)



    rect = cv2.minAreaRect(cnt)

    box = cv2.boxPoints(rect)

    box = np.int0(box)



    # Check box rotation using diagonal vectors

    vectors = np.array([box[(i+2)%4]-box[i] for i in range(len(box))])

    angles = np.arctan2(*vectors.T)

    b0 = np.argmin(np.pi - angles)

    b2 = (b0 + 2) % 4

    b1 = np.argmin(np.pi + angles)

    

    pts1 = box[[b0, b1, b2]].astype(np.float32)

    pts2 = np.array([[w+h_border, h+v_border], [h_border, h+v_border], [h_border, v_border]], dtype=np.float32)



    W = cv2.getAffineTransform(pts1, pts2)

    roi = cv2.warpAffine(im, W, (w + 2*h_border, h + 2*v_border), flags=cv2.INTER_AREA)

    roi_mask = cv2.warpAffine(mask, W, (w + 2*h_border, h + 2*v_border), flags=cv2.INTER_AREA)

    return roi, roi_mask
%matplotlib inline



rnd_images_with_mask = ['3_103', '4_57', '5_107', '7_2', '9_104', '10_100', '13_44', '15_78',

                        '23_35', '30_100', '36_95', '40_31', '41_82', '44_67', '47_108']



fig, axs = plt.subplots(5, 3, figsize=(15, 15))

axs = axs.ravel()

for im_mask, ax in zip(rnd_images_with_mask, axs):

    ax.imshow(np.c_[focus(im_mask)], cmap='gray')

    ax.set_title(im_mask)

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)