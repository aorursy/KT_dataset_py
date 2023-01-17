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
with open("../input/train.json") as f:
   data = json.load(f)

print("Dies sind die Schlüssel im Dictionary")
print(data[0].keys())


Y = []
for element in data:
    Y.append(element["class"])
from matplotlib import pyplot as plt
plt.hist(Y,max(Y))
print("es gibt insgesammt: "+ str(max(Y) + 1) + " Klassen (inkl. Klasse 0)")
# Laden der Bilder

from PIL import Image
X = []
shapes = []
for element in data:
    X.append(np.array(Image.open("../input/train/train/"+ element["filename"])))
    shapes.append(X[-1].shape)
    

    

import cv2
# resizen eines bildes
X_resized = cv2.resize(X[10],(64,64))
plt.imshow(X_resized)