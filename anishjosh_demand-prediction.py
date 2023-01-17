# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output.
print("train has {} rows and {} columns".format(train.shape[0],train.shape[1]))
print("FEATURES:")
print(train.columns)
def item_loc(id_,feat=None):  #the function helps on selective feature searching
    if id_==int(id_) and feat==None:
        nil_feat = train.iloc[id_,:]
    else:
        nil_feat = train.iloc[id_,:][int(feat)]
    return nil_feat
from zipfile import ZipFile
image_dir = ZipFile("../input/train_jpg.zip")
filenames = image_dir.namelist()[1:200]
def get_blurrness(file): #extracting blurness
    exfile = image_dir.read(file)
    arr = np.frombuffer(exfile, np.uint8)
    if arr.size > 0:   # exclude dirs and blanks
        imz = cv2.imdecode(arr, flags=cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(imz, cv2.CV_64F).var()
    else: 
        fm = -1
    return fm
blurrness = []
files = filenames[60:75]
for i in range(0, len(files)):
    print(i)
    blurrness.append(get_blurrness(files[i]))
    
blurrness
x = 60 #
for i,p in zip(range(len(blurrness)),range(x,x+len(blurrness))):
    print(blurrness[i],item_loc(p,17))
