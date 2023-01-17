

get_ipython().magic(u'matplotlib inline')

# https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py

import utils; reload(utils)

from utils import *

from __future__ import division, print_function





path = 'train/'

batch_size=32





batches = get_batches(path+'', gen=image.ImageDataGenerator(), batch_size=batch_size, shuffle=False, class_mode=None)



filenames = batches.filenames

raw_filenames = [f.split('/')[-1] for f in filenames]



from collections import Counter

dup_filenames = []

for item, count in Counter(valid_filenames).iteritems():

    if count > 1:

        dup_filenames.append(item)



np.savetxt('raw_filenames.txt',raw_filenames, fmt="%s")





import numpy as np

from scipy import misc

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import pandas as pd

import cv2

import os

from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors



# https://www.kaggle.com/xenocide/the-nature-conservancy-fisheries-monitoring/fishy-neighbors-knn-solution-log-loss-1-65074

def extract_color_histogram(image,bins = (8,8,8)):

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,256,0,256,0,256])

    cv2.normalize(hist,hist)

    return hist.flatten()





hists = []

for i,image_path in enumerate(filenames):

    image = misc.imread(path+image_path)

    image = cv2.resize(image,(128,128)) # (256,256)

    hist = extract_color_histogram(image)

    hists.append(hist)

    if (i+1)%1000 == 0:

        print (i)





kmeans = KMeans(n_clusters=1000, random_state=0).fit(hists)

kmeans.cluster_centers_.shape

x=np.where(np.array(hists)==kmeans.cluster_centers_[:1])



valid_filenames=[]

for i in range(1000):

    d = kmeans.transform(hists)[:, i]

    ind = np.argsort(d)[::][:1]

    valid_filenames.append(filenames[ind[0]])

    if (i+1)%100 == 0:

        print (i+1)



valid_filenames=[]

for i in range(1000):

    ind = np.random.choice(np.where(kmeans.labels_==i)[0],8) # replace=False not being used

    for t in ind:

        valid_filenames.append(filenames[t])



np.savetxt('valid_filenames.txt',np.array(valid_filenames), fmt="%s")





get_ipython().magic(u'mkdir data')

get_ipython().magic(u'mkdir -p data/train')

get_ipython().magic(u'mkdir -p data/valid')

get_ipython().magic(u'mkdir -p data/train/train-jpg')

get_ipython().magic(u'mkdir -p data/valid/train-jpg')





from shutil import copyfile

for i,fn in enumerate(filenames):

    if fn in valid_filenames:

        copyfile(path+fn, 'data/valid/'+fn)

    else:

        copyfile(path+fn, 'data/train/'+fn)

    if (i+1)%1000 == 0:

        print (i+1)