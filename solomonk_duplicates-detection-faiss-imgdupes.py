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
! conda install -y faiss-gpu cudatoolkit=10.0 -c pytorch
!  pip install imgdupes
! ls -la ../input/
! imgdupes --recursive  --faiss-flat "../input/test_images" phash 1
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib import rcParams



%matplotlib inline



# figure size in inches optional

rcParams['figure.figsize'] = 15 ,15

def plotTwo(img_A,img_B):

    # read images    

    # display images

    fig, ax = plt.subplots(1,2)

    ax[0].imshow(img_A);

    ax[1].imshow(img_B);
img_A = mpimg.imread('../input/test_images/01c31b10ab99.png')

img_B = mpimg.imread('../input/test_images/b29bd35acaf6.png')

plotTwo (img_A,img_B)
img_A = mpimg.imread('../input/test_images/417d3908ee21.png')

img_B = mpimg.imread('../input/test_images/9d9de8c9afb5.png')

plotTwo (img_A,img_B)