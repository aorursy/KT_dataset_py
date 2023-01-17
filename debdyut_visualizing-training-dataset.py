# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import shutil

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import random

import matplotlib.image as mpimg



%matplotlib inline
df = pd.read_csv('../input/labels.csv')



df.head()
headers = list(df)



print(headers)
# Number of images per class

gr_labels = df.groupby("breed").count()

gr_labels = gr_labels.rename(columns = {"id" : "count"})

images_count = list(gr_labels['count'])



gr_labels.sort_values("count", ascending=False).head()
gr_labels.sort_values("count", ascending=False).tail()
classes = np.unique(df["breed"]).reshape(120, 1)



avg = sum(images_count) / float(len(images_count))

colors = []

for i in range(0, len(images_count)):

    images_count[i] = images_count[i] - avg

    if images_count[i] < 0:

        colors.append('r')

    else:

        colors.append('b')

        

plt.figure(figsize=(20,10))



y_pos = np.arange(len(classes))

plt.axhline(y = 0,color='grey') 

plt.bar(y_pos, images_count, align='center', color = colors, alpha=0.5, width = 0.3)

plt.ylabel('Deviation from average value')

plt.xlabel('Classes')

plt.title('Distribution of training images')



for i, v in enumerate(images_count):

    if v < 0:

        yd = -1.5

    else:

        yd = 1.5

    n = int(v + avg)

    plt.text(i-0.75, v + yd, str(n), color='blue')



plt.show()