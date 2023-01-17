from matplotlib.pyplot import imshow

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

import pandas as pd

import numpy as np

import cv2

import os



print(os.listdir('../input/'))
train = pd.read_csv('../input/understanding_cloud_organization/train.csv')
train.head()
train.dtypes
Image_Label = train.pop('Image_Label')



train['label'] = Image_Label.apply(lambda x: x.split('_')[1])

train['image'] = Image_Label.apply(lambda x: x.split('_')[0])
train.isnull().sum()
train.head()
sns.countplot(train['label'])

plt.plot()
path = '../input/understanding_cloud_organization/train_images/'



labels = train['label'].values





for i, img in enumerate(train['image'][:4]):

    image = plt.imread(path + img)

    plt.imshow(image)

    plt.title(labels[i])

    plt.show()
def decode(mask, shape=(1400, 2100)):

    

    m = mask.split()

    a = list()

    

    for x in (m[0:][::2], m[1:][::2]):

        a.append(np.asarray(x, dtype=int))

    

    starts, lengths = a

    starts -= 1

    stop = starts + lengths

    

    image = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for i, j in zip(starts, stop):

        image[i:j] = 1

        

    image = image.reshape(shape, order='F') 

    return image
plt.figure(figsize=[60, 30])



for i, row in train[:16].iterrows():

    img = cv2.imread(path +  row['image'])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    enc_pix = row['EncodedPixels']

    try:

        mask = decode(enc_pix)

    except:

        mask = np.zeros((1400, 2100))

        

    plt.subplot(4, 4, i+1)

    plt.imshow(img)

    plt.imshow(mask, alpha=0.6, cmap='gray')

    plt.title("Label %s" % row['label'], fontsize=32)

    plt.axis('off')    

    

plt.show()