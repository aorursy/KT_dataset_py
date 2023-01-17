from pycocotools import coco



import numpy as np

import skimage.io as io

import matplotlib.pyplot as plt

import pylab

import random

import os

pylab.rcParams['figure.figsize'] = (8.0, 6.0)



modalities = ['rgb', 'thermal', 'depth']



cocoGt = dict()



if '/kaggle/input' not in os.getcwd():

    os.chdir('../input/')



for modality in modalities:

    annFile = './trimodal-' + modality + '.json'

    cocoGt[modality] = coco.COCO(annFile)

    
imgIds = cocoGt['rgb'].getImgIds()

chosenImgId = 1500 # random.randint(0, 5721)



# Show annotations for chosen image

for modality in modalities:

    annIds = cocoGt[modality].getAnnIds(imgIds=[chosenImgId])

    

    if len(annIds) == 0:

        print('No annotations for image ' + imgPath['file_name'])

        break

    

    anns = cocoGt[modality].loadAnns(annIds)

    imgPath = cocoGt[modality].loadImgs([chosenImgId])[0]

    

    print(imgPath['file_name'])

    

    datasetPath = './trimodaldataset/'

    img = io.imread(datasetPath + imgPath['file_name'])

    

    plt.figure()

    plt.axis('off')

    plt.imshow(img)

    cocoGt[modality].showAnns(anns)