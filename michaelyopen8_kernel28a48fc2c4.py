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
train_images = os.listdir("../input/train")
print(len(train_images))
test_images = os.listdir("../input/test")
print(len(test_images))
train_images[:10]
print(os.listdir("../input/train")[:10])
df = pd.read_csv('../input/train.csv')
i = 5
for row in df.iterrows():
    print(row)
    i -= 1
    if i == 0:
        break
from PIL import Image
train_img = []
for image_path in train_images:
    img = Image.open('../input/train/' + image_path).convert('L').resize((128,128))
    train_img.append(np.array(img))
shape_set = {t.shape for t in train_img}
shape_set

from collections import defaultdict
data_distribution = defaultdict(int)
#print(len(df.Id.unique()))
for row in df.iterrows():
    data_distribution[row[1].Id] += 1
#print(data_distribution)

spread = defaultdict(int)
for k, v in data_distribution.items():
    spread[v] += 1
from pprint import pprint
pprint(spread)
spread_named = defaultdict(int)
sorted([(v, k) for k, v in data_distribution.items() if v > 20])


import matplotlib.pylab as plt
fig = plt.figure(num=None, figsize=(30, 30))
for i in range(1, 10):
    fig.add_subplot(3,3,i)
    plt.imshow(train_img[i])
plt.show()
