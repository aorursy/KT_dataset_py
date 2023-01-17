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
SIZE = 512

max_shapes_of_one_kind = 10

size_ = {}

size_['sq'] = (20, 30)

size_['trig'] = (20, 30)

size_['circ'] = (5,10)
import cv2

import random

import matplotlib.pyplot as plt

%matplotlib inline





img = np.zeros((512,512,3), 'int8')

for shape_ in size_:

    n = random.randint(0, max_shapes_of_one_kind)

    for j in range(n):

        if shape_ == 'sq':

            s = random.randint(size_[shape_][0], size_[shape_][1])

            x = random.randint(0, SIZE-s)

            y = random.randint(0, SIZE-s)

            cv2.rectangle(img, (x, y), (x+s, y+s), (255,255,255), thickness=cv2.FILLED)

        elif shape_ == 'circ':

            s = random.randint(size_[shape_][0], size_[shape_][1])

            x = random.randint(0, SIZE-s//2)

            y = random.randint(0, SIZE-s//2)

            cv2.circle(img, (x, y), s//2, (255,255,255), thickness=cv2.FILLED)

        elif shape_ == 'trig':

            s = random.randint(size_[shape_][0], size_[shape_][1])

            cx = random.randint(0, SIZE-s)

            cy = random.randint(0, SIZE-s)

            p1 = (cx, cy+s)

            p2 = (cx+s, cy+s)

            p3 = (cx+s//2, cy)

            

            cv2.fillConvexPoly(img, np.array((p1, p2, p3), dtype = int), (255,255,255))

            

            

            

plt.imshow(img)

plt.figure()
