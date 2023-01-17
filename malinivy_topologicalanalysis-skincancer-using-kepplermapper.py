# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread

import cv2
mal_images = glob('../input/skin-cancer-malignant-vs-benign/data/train/malignant/*')
ben_images = glob('../input/skin-cancer-malignant-vs-benign/data/train/benign/*')
len(mal_images)
mal_images[0:5]
print(len(ben_images))
ben_images[0:5]
benign=pd.DataFrame()
labels = []
for imagePath in ben_images:
  column_name=imagePath.split('/')[-1].split('.')[0]
  image=cv2.imread(imagePath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  benign[column_name]= image.flatten()
  labels.append('0')
benign.shape
benign=benign.transpose()

benign['label']=labels
labels_2 = []
malignant=pd.DataFrame()

for imagePath in mal_images:
  column_name=imagePath.split('/')[-1]
  image=cv2.imread(imagePath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # image=cv2.resize(image, (300, 300),interpolation=cv2.INTER_AREA)
  malignant[column_name]=image.flatten()
  labels_2.append('1')
malignant=malignant.transpose()
malignant['label']=labels_2
df = pd.concat([benign, malignant])
df['label'].value_counts()
Xtrain=df.drop(['label'],axis=1)
!pip install kmapper
import kmapper as km

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper_full = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data_full = mapper_full.fit_transform(Xtrain,
                                      projection=sklearn.manifold.TSNE(perplexity=50))


# Create the graph (we cluster on the projected data and suffer projection loss)
graph_full = mapper_full.map(projected_data_full,
                   clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
                   cover=km.Cover(35, 0.4))
Y=df['label']
# Matplotlib examples
km.draw_matplotlib(graph_full)
plt.show()
# Tooltips with the target y-labels for every cluster member
mapper_full.visualize(graph_full,
                 title="Skin Cancer Mapper with  Labels ",
                 path_html="/kaggle/working/skin_cancer_ylabel_images.html",
                 custom_tooltips=Y)



