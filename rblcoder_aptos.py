import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/train.csv')
df_train.info()
df_train.head()
df_train.diagnosis.value_counts()
df_train.diagnosis.nunique()
df_test = pd.read_csv('../input/test.csv')
df_test.info()
df_test.head()
df_sample_submission = pd.read_csv('../input/sample_submission.csv')
df_sample_submission.info()
df_sample_submission.head()
len(os.listdir('../input/train_images/'))
os.listdir('../input/train_images/')[:5]
len(os.listdir('../input/test_images/'))
os.listdir('../input/test_images/')[:5]
import imageio
imageio.imread('../input/train_images/' + os.listdir('../input/train_images/')[0]).shape
imageio.imread('../input/test_images/' + os.listdir('../input/test_images/')[0]).shape
_ = plt.imshow(imageio.imread('../input/train_images/' + os.listdir('../input/train_images/')[0]))
_ = plt.imshow(imageio.imread('../input/test_images/' + os.listdir('../input/test_images/')[0]))
#https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas

df_train_sample = df_train.groupby('diagnosis', group_keys=False).apply(lambda x: x.sample(min(len(x), 2)))
df_train_sample
_ = plt.imshow(imageio.imread('../input/train_images/' + df_train_sample.iloc[4,0] + '.png'))
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,20))

i = 0

images = []

labels = []

for ar in axes:

    for ac in ar:

        image = imageio.imread('../input/train_images/' + df_train_sample.iloc[i,0] + '.png')

        ac.imshow(image)

        images.append(image)

        label = str(df_train_sample.iloc[i,1])

        ac.set_title(label)

        labels.append(label)

        i += 1

#https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,20))

i = 0

for ar in axes:

    for ac in ar:

        ac.hist(images[i].ravel(), bins=32, range=[0, 256])

        ac.set_title(labels[i])

        i += 1
#Learning from https://www.datacamp.com/courses/biomedical-image-analysis-in-python

from scipy import ndimage 

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,20))

i = 0

for ar in axes:

    for ac in ar:

        #print(ndimage.histogram(images[i].ravel(), min=0, max=255, bins=256))

        ac.plot(ndimage.histogram(images[i].ravel(), min=0, max=255, bins=256))

        ac.set_title(labels[i])

        i += 1
#Learning from https://www.datacamp.com/courses/biomedical-image-analysis-in-python

from scipy import ndimage 

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,20))

i = 0

for ar in axes:

    for ac in ar:

        #print(ndimage.histogram(images[i].ravel(), min=0, max=255, bins=256))

        histogram_calc = ndimage.histogram(images[i].ravel(), min=0, max=255, bins=256)

        ac.plot(histogram_calc.cumsum()/histogram_calc.sum())

        ac.set_title(labels[i])

        i += 1
#https://github.com/aleju/imgaug/issues/66

from imgaug import augmenters as iaa



# aug1 = iaa.GaussianBlur(sigma=(0, 2.0))

# aug2 = iaa.AdditiveGaussianNoise(scale=0.01 * 255)



aug3=iaa.SimplexNoiseAlpha(

   first=iaa.EdgeDetect(1.0),

   second=iaa.ContrastNormalization((0.5, 2.0)),

   per_channel=0.5

)



def additional_augmenation(image):

  #  image = aug1.augment_image(image)

    image = aug3.augment_image(image)

    return image



fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,20))

i = 0

for ar in axes:

    for ac in ar:

        ac.imshow(additional_augmenation(images[i]))

        ac.set_title(labels[i])

        i += 1
