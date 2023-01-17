# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/train"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib as mpl

import matplotlib.pyplot as plt

import os





names3 = [None, None, None]

for i in range(3):

    names3[i] = os.listdir(f'../input/train/Type_{i+1}/')[:100]





#im = plt.imread(os.path.join('../input/train/Type_1/', type1_names[0]))

#plt.imshow(im)

#plt.subplot

import matplotlib as mpl

import matplotlib.pyplot as plt

import random



BASE_PATH = '../input/'



ims_per_type = 4

# ensure enough space to show large(r) images

plt.figure(figsize=(12, 12))

for i in range(3):

    for j in range(ims_per_type):

        # plot_number increments across rows first

        plt.subplot(3, ims_per_type, i * ims_per_type + j + 1)

        basename = names3[i][random.randrange(0,100)]

        fn = os.path.join(BASE_PATH, f'train/Type_{i+1}', basename)

        print(i, j, fn)

        plt.imshow(plt.imread(fn))

        plt.xticks([])

        plt.yticks([])

        plt.title(f't{i+1} - {basename}')



# kill h and w space to make more space for images

plt.subplots_adjust(hspace=.001, wspace=0.001)

plt.tight_layout()
