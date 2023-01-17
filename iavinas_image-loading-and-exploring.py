import numpy as np

import pandas as pd

import seaborn as sns

from pathlib import Path



import matplotlib.pyplot as plt

from IPython.display import display

from PIL import Image

%matplotlib inline
train_df = pd.read_csv("../input/train_relationships.csv")

test_df = pd.read_csv("../input/sample_submission.csv")
train_df.head()
len(train_df)
test_data = np.random.beta(a=1,b=1 , size= (100, 100, 3) )

plt.imshow(test_data)
import glob

path = '../input/train/' + train_df.p1[100]

image_datas = [Image.open(f) for f in glob.glob(path + "/*.jpg", recursive=True)]
image_datas
f, ax = plt.subplots(1,4 ,  figsize=(50,20))

for i in range(4):

    ax[i].imshow(image_datas[i])
path = '../input/train/' + train_df.p2[100]

image_datas = [Image.open(f) for f in glob.glob(path + "/*.jpg", recursive=True)]

image_datas
f, ax = plt.subplots(1,4 ,  figsize=(50,20))

for i in range(4):

    ax[i].imshow(image_datas[i])