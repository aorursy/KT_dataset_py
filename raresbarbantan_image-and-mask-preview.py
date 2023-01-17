

import os

import cv2

from matplotlib import pyplot as plt

from PIL import Image



%matplotlib inline



train_images = [x for x in os.listdir('../input/train') if x.endswith('.jpg')]



fig = plt.figure(figsize=(20, 30))



for i,f in enumerate(train_images[:5]):  # look at first 5 

    car = f.split('.')[0]

    img = cv2.imread('../input/train/{0}.jpg'.format(car), 1)

    mask = Image.open('../input/train_masks/{0}_mask.gif'.format(car))





    fig.add_subplot(5, 2, 2*i+1)

    plt.imshow(img)



    fig.add_subplot(5, 2, 2*i+2)

    plt.imshow(mask)