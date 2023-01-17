# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as nm

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import cv2 as cv2

import os


allFeatures=[]



for filename in os.listdir('../input/'):

    img = cv2.imread('../input/'+filename)

    img1 = cv2.resize(img, (240,240), interpolation = cv2.INTER_AREA)

    mask = nm.zeros(img1.shape[:2],nm.uint8)

    bgdModel = nm.zeros((1,65),nm.float64)

    fgdModel = nm.zeros((1,65),nm.float64)

    rect = (5,5,235,235)

    cv2.grabCut(img1,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)

    mask2 = nm.where((mask==2)|(mask==0),0,1).astype('uint8')

    img1 = img1*mask2[:,:,nm.newaxis]

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    #img1= cv2.GaussianBlur(img1,(5,5),cv2.BORDER_DEFAULT)

    ll=cv2.equalizeHist(img1)

    orb = cv2.ORB_create(nfeatures=200)

    keypoints, descriptors = orb.detectAndCompute(ll, None)

    allFeatures.append(descriptors)#array[ImageNb][FeatureNb]
kmeans = KMeans(n_clusters = 200, n_init=10, init='random')

gg=[item for sublist in allFeatures for item in sublist]

kmeans.fit(gg)#vocabulary
documentsPerFeature=[0] * 200



alla=[]

for i in allFeatures:

    clusters=[]

    for u in i:

        cluster=kmeans.predict([u])  

        clusters.append(cluster[0])

        

    nums=pd.Series(clusters).value_counts()

    tf=nums.apply(lambda a:a/nums.sum())

    alla.append(tf.to_dict())

#tf for each feature in the document



for i in alla:

    for k in i:

        documentsPerFeature[k-1]=documentsPerFeature[k-1]+1

#number of documents containing a given feature
import math as m

fr= pd.DataFrame(columns= range(1,201))

for i in alla:

    row=[0]*200

    totalWords=0

    for k in i:

        idf=m.log(len(alla)/documentsPerFeature[k-1])

        res=i[k]*idf

        row[k]=res

    fr=fr.append(pd.Series(row, range(1,201)),ignore_index=True )  

fr       

    