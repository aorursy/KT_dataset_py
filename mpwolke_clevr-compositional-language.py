#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTYtVeDahbFq1pHowTuOIOvGkxf-4bisfNPDpSgMsf0JAyv3Pnl',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
clevr_file = '../input/clevr-dataset/CLEVR_v1.0/COPYRIGHT.txt'

with open(clevr_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
clevr_file = '../input/clevr-dataset/CLEVR_v1.0/README.txt'

with open(clevr_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
clevr_file = '../input/clevr-dataset/CLEVR_v1.0/LICENSE.txt'

with open(clevr_file) as f: # The with keyword automatically closes the file when you are done

    print (f.read(1000))
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/clevr-dataset/CLEVR_v1.0/images/train/CLEVR_train_049804.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)
#codes from Endi Niu @niuddd

img = cv2.imread('/kaggle/input/clevr-dataset/CLEVR_v1.0/images/train/CLEVR_train_052823.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(8,8))

plt.imshow(img)