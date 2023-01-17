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
import json
with open('../input/train_xy.json') as f:
    data = json.load(f)
# print the data one time so we are sure that it has loaded correctly
print(data[0])
from PIL import Image
from tqdm import tqdm
import numpy as np

# load the data 
X = []
Y_classes = []
Y_centroids = []

#print(os.listdir("../input/train/train/"))

# a very simple load of data

for entry in tqdm(data):
    img_pil = Image.open("../input/train/train/" + entry['filename'])
    img = np.array(img_pil)
    
    X.append(img)
    Y_classes.append(entry['class'])
    Y_centroids.append([entry['x_centroid'],entry['y_centroid']])
    
    
    
