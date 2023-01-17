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
# Laden der json file
with open("../input/train.json") as f:
    data = json.load(f)

one_image_per_class = []

cnt = 0
for element in data:
    
    if element['class'] == cnt:
        one_image_per_class.append(element)
        cnt += 1
    
from PIL import Image
images = []
for element in one_image_per_class:
       images.append(Image.open("../input/train/train/" + element["filename"]))
from matplotlib import pyplot as plt
for Class,image in enumerate(images):
    plt.figure(Class)
    plt.imshow(np.array(image))
    plt.title(Class)