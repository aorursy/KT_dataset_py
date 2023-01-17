import numpy as np

import matplotlib.pyplot as plt



import bson

import cv2 



from skimage.data import imread

from io import BytesIO
def squeeze_image(img):

    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    

    all_white = 255 * img.shape[0]

    

    proj_x = np.where(greyscale_img.sum(axis=0) != all_white)[0] 

    proj_y = np.where(greyscale_img.sum(axis=1) != all_white)[0] 

    

    squeezed_image = img[proj_y[0]:proj_y[-1]+1, proj_x[0]:proj_x[-1]+1]

    

    return squeezed_image
data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))

small_data = [next(data) for _ in range(10)]

for d in small_data:

    for i, pic in enumerate(d['imgs']):

        img_bytes = BytesIO(pic['picture'])

        img = imread(img_bytes)

        

        plt.subplots()

        plt.subplot(1, 2, 1)

        plt.imshow(img)

        plt.subplot(1, 2, 2)

        plt.imshow(squeeze_image(img))