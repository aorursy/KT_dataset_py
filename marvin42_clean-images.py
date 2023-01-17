from PIL import Image

import cv2

import numpy as np

from tqdm import tqdm

from skimage.morphology import skeletonize

import os



def get_data(n):

    image_id = train[n][0]

    image, raw_labels = Image.open(image_path(image_id)), train[n][1]

    image = image.convert('L')

    image = np.array(image, dtype=np.uint8)

    return image_id, image, parse_labels(raw_labels)



def threshold(image, thinning=False, soft=False):

    ret,im = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    im = 255 - im

    iters = 2 if soft else 4

    kernel = np.ones((2,2),np.uint8)

    im = cv2.dilate(im,kernel,iterations = iters)

    im = cv2.erode(im,kernel,iterations = iters)

    if thinning:

        im = skeletonize(im/255)

        im = np.asarray(im*255, dtype=np.uint8)

        im = cv2.dilate(im,kernel,iterations = 2)

    return im



def parse_labels(labels):

    label_parts = labels.split()

    res = []

    for r1, r2 in zip(range(0, len(label_parts), 5), range(5, len(label_parts), 5)):

        codepoint, x, y, w, h = label_parts[r1:r2]

        x,y,w,h = map(int, [x,y,w,h])

        res.append((codepoint, x,y,w,h))

    return res



def get_crop(image, label):

    c,x,y,w,h = label

    crop = image[y:y+h,x:x+w]

    return crop



def clean(image, labels):

    tmp_image = threshold(image, True, True)

    zeros = np.zeros(tmp_image.shape, dtype=np.uint8)

    for label in labels:

        c,x,y,w,h = label

        crop = get_crop(tmp_image, label)

        zeros[y:y+h, x:x+w] = crop

    return 255 - zeros



def get_codepoint(label):

    return label[0]



def show(*images, size=400):

    prev = np.hstack(images)

    timg = Image.fromarray(prev)

    timg.thumbnail((size*len(images),size), Image.ANTIALIAS)

    return timg



train = [x.split(",") for x in open("../input/train.csv").read().strip().split("\n")[1:]]

image_path = lambda image_id : "../input/train_images/" + image_id + ".jpg"



index = 10

image_id, image, labels = get_data(index)

cleaned_image = 255 - threshold(image, labels)

show(cleaned_image, size=800)