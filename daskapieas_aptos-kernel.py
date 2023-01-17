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
import pandas as pd

import matplotlib.pyplot as plt

import cv2
train = pd.read_csv('../input/train.csv')

train.head(5)
X = train['id_code']

y = train['diagnosis']

print(y.value_counts())



y.hist()
X.iloc[0]
img = plt.imread("../input/train_images/"+X.iloc[0]+".png")

plt.imshow(img)

print(img.shape)
print(len(y.unique()))

print(len(y))
SEED = 77

IMG_SIZE = 512

fig = plt.figure(figsize=(25, 16))

# display 10 images from each class

for class_id in sorted(y.unique()):

    for i, (idx, row) in enumerate(train.loc[train['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):

        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])

        path = f"../input/train_images/{row['id_code']}.png"

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))



        plt.imshow(image)

        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['id_code']) )

train.loc[train['diagnosis'] == 0].sample(5)