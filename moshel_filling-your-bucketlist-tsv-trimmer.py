# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv
#load the image list
train_images = {}
with open("train-images-boxable-with-rotation.csv", "r") as f:
    for l in f:
        r = l.split(",")
        if r[0] == 'ImageID':
            continue
        train_images[r[2]] = r[0]
            
print("loaded {} image ids".format(len(train_images)))
    
!wget https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train0.tsv

#screen the images to include only the ones in the training set
infn = "open-images-dataset-train0.tsv"
count = 0
outf = open("trimmed_"+infn, "w+")

with open(infn, "r") as f:
    for l in f:
        r = l[:-1].split("\t")
        if len(r) == 1: #header
            outf.write(l)
            continue
        try:
            id = train_images[r[0]]
            outf.write(l)
            count += 1
        except:
            continue
#        print(r[2])
outf.close()
print(count)

!head trimmed*