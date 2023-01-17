import numpy as np 

import pandas as pd

import os

import cv2

import random

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from keras.utils import to_categorical
imagesPath = '/kaggle/input/utkface-images/utkfaceimages/UTKFaceImages/'

labelsPath = '/kaggle/input/utkface-images/'
files = os.listdir(labelsPath)

labels = pd.read_csv(labelsPath+files[0])
images = os.listdir(imagesPath)
labels.head()
labels.sample(10)
labels.describe()
labels.dtypes
print("Unique values in gender; ", labels['gender'].unique())
print("Unique values in ethnicity; ", labels['ethnicity'].unique())
ages = labels['age'].unique()

ages.sort()

print("Unique values in age; ", ages)
labels = labels[labels.ethnicity != '20170109150557335.jpg.chip.jpg']

labels = labels[labels.ethnicity != '20170116174525125.jpg.chip.jpg']

labels = labels[labels.ethnicity != '20170109142408075.jpg.chip.jpg']



labels = labels.astype({'ethnicity': 'int64'})
labels.describe()
labels.dtypes
plt.figure(figsize=(13,8))

labels['age'].hist(bins=len(ages));
labels.groupby('age').count().sort_values('image_id', ascending=False).head(10)
plt.figure(figsize=(13,8))

labels['age'].hist(bins=[0, 5, 18, 24, 26, 27, 30, 34, 38, 46, 55, 65, len(ages)]);
genders = labels['gender'].unique()



plt.figure(figsize=(13,8))

labels['gender'].hist(bins=len(genders));
ethnicity = labels['ethnicity'].unique()



plt.figure(figsize=(13,8))

labels['ethnicity'].hist(bins=len(ethnicity));
def show_images(images, cols = 1, titles = None):

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: print('Serial title'); titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.imshow(image, cmap=None)

        a.set_title(title, fontsize=50)

        a.grid(False)

        a.axis("off")

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.savefig('faceless.png')

plt.show()
samples = np.random.choice(len(images), 16)

sample_images = []

sample_labels = []

for sample in samples:

    img = cv2.imread(imagesPath+images[sample])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    sample_images.append(img)
show_images(sample_images, 4)