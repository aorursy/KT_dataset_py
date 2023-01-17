import os

import numpy as np

import cv2

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
# create list of all directories

base_dir = '/kaggle/input/alaska2-image-steganalysis/'

image_dirs = ['Cover','JUNIWARD', 'JMiPOD',  'UERD']
data = {}



for image_dir in image_dirs:

    if image_dir=="Test":

        continue

    images = []

    for file in os.listdir(os.path.join(base_dir, image_dir)):

        images.append(file)

    data[image_dir]=images



train_df = pd.DataFrame(data)
train_df.head()
def similarity_test(df):

    similarity = set(df['Cover'] == df['JUNIWARD'])

    similarity = similarity | set(df['Cover'] ==df['JMiPOD'])

    similarity = similarity | set(df['Cover'] ==df['UERD'])

    return similarity

similarity_test(train_df)
train_df.describe()
train_df.info()
image_id_1 = 41731

image_id_2 = 12314

image_id_3 = 28962

image_id_4 = 127

sample_images_1 = [base_dir + x[0] +'/'+x[1] for x in zip(list(train_df.columns) , list(train_df.iloc[image_id_1,:]))]

sample_images_2 = [base_dir + x[0] +'/'+x[1] for x in zip(list(train_df.columns) , list(train_df.iloc[image_id_2,:]))]

sample_images_3 = [base_dir + x[0] +'/'+x[1] for x in zip(list(train_df.columns) , list(train_df.iloc[image_id_3,:]))]

sample_images_4 = [base_dir + x[0] +'/'+x[1] for x in zip(list(train_df.columns) , list(train_df.iloc[image_id_4,:]))]



sample_images_1
_, axs = plt.subplots(1, 4, figsize=(12, 12))

axs = axs.flatten()

for img,ax in zip(sample_images_1,axs):

    ax.imshow(cv2.imread(img))

    ax.set_title(img.split('/')[-2])

plt.show()
multiplier = 10000 # This is used to brighten the diffrential image



_, axs = plt.subplots(1, 4, figsize=(12, 12))

axs = axs.flatten()



Cover = np.array(cv2.imread(sample_images_1[0]))

for img,ax in zip(sample_images_1, axs):

    if 'Cover' in img:

        ax.imshow(Cover)

        ax.set_title('Cover')

        continue

    image = np.array(cv2.imread(img))

    new_image = (Cover - image)*multiplier

    ax.imshow(new_image)

    title = img.split('/')[-2] + ' differential'

    ax.set_title(title)
_, axs = plt.subplots(1, 4, figsize=(12, 12))

axs = axs.flatten()



Cover = np.array(cv2.imread(sample_images_2[0]))

for img,ax in zip(sample_images_2, axs):

    if 'Cover' in img:

        ax.imshow(Cover)

        ax.set_title('Cover')

        continue

    image = np.array(cv2.imread(img))

    new_image = (Cover - image)*multiplier

    ax.imshow(new_image)

    title = img.split('/')[-2] + ' differential'

    ax.set_title(title)
_, axs = plt.subplots(1, 4, figsize=(12, 12))

axs = axs.flatten()



Cover = np.array(cv2.imread(sample_images_3[0]))

for img,ax in zip(sample_images_3, axs):

    if 'Cover' in img:

        ax.imshow(Cover)

        ax.set_title('Cover')

        continue

    image = np.array(cv2.imread(img))

    new_image = (Cover - image)*multiplier

    ax.imshow(new_image)

    title = img.split('/')[-2] + ' differential'

    ax.set_title(title)
from skimage.feature import hog



def hog_image(img):

    img=img[:,:,1] 

    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(2, 2), visualize=True)    

    return hog_image
_, axs = plt.subplots(1, 4, figsize=(12, 12))

axs = axs.flatten()

for img,ax in zip(sample_images_4,axs):

    ax.imshow(hog_image(np.array(cv2.imread(img))))

    ax.set_title(img.split('/')[-2])

plt.show()
_, axs = plt.subplots(1, 4, figsize=(12, 12))

axs = axs.flatten()



Cover = np.array(cv2.imread(sample_images_4[0]))

for img,ax in zip(sample_images_4, axs):

    if 'Cover' in img:

        ax.imshow(hog_image(Cover))

        ax.set_title('Cover')

        continue

    image = np.array(cv2.imread(img))

    new_image = (Cover - image)

    ax.imshow(hog_image(new_image))

    title = img.split('/')[-2] + ' differential'

    ax.set_title(title)