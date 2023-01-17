import numpy as np

import pandas as pd



df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
df.head(10)
def get_id(s):

    s = s.split('_')

    return 'ID_' + s[1]



def get_hemorrhage_type(s):

    s = s.split('_')

    return s[2]

    

df['Image_ID'] = df.ID.apply(get_id)

df['Hemhorrhage_Type'] = df.ID.apply(get_hemorrhage_type)



df.set_index('Image_ID')
df.head(10)
from os import listdir

from os.path import isfile, join

from pathlib import Path



# To read medical images.

import pydicom



import matplotlib.pyplot as plt



train_images_dir = Path('../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/')

train_images = [str(train_images_dir / f) for f in listdir(train_images_dir) if isfile(train_images_dir / f)]
train_images[0]
ds = pydicom.dcmread(train_images[0])

im = ds.pixel_array



plt.imshow(im, cmap=plt.cm.gist_gray);
fig=plt.figure(figsize=(15, 10))

columns = 5; rows = 4

for i in range(1, columns*rows +1):

    ds = pydicom.dcmread(train_images[i])

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap=plt.cm.gist_gray)

    fig.add_subplot
import seaborn as sns



sns.countplot(df.Label)