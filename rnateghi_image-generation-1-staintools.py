!pip install  -qq ../input/staintools/spams-2.6-cp37-cp37m-linux_x86_64.whl
!pip install  -qq ../input/staintools/staintools-2.1.0-py3-none-any.whl
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import PIL

from IPython.display import Image, display

from tqdm.notebook import tqdm

import cv2

from skimage import io

import spams

import staintools

from staintools.reinhard_color_normalizer import ReinhardColorNormalizer

from PIL import Image

import os

import shutil

print(os.listdir('/kaggle/input/prostate-cancer-grade-assessment'))

import zipfile





# Any results you write to the current directory are saved as output.
# Location of the training images

data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/'



# Location of training labels

pds = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')

print(pds.head())



sample = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')

OUT_TRAIN = 'train.zip'
IMG_SIZE = 256

SEQ_LEN = 36

SLIDE = 1

Scaling = 1

def get_patches_train(img_path, num_patches, img_size):

    p_size = img_size

    img = io.MultiImage(img_path)[SLIDE] / 255

    pad0, pad1 = (p_size - img.shape[0] % p_size) % p_size, (p_size - img.shape[1] % p_size) % p_size

    img = np.pad(

        img,

        [

            [pad0 // 2, pad0 - pad0 // 2], 

            [pad1 // 2, pad1 - pad1 // 2], 

            [0, 0]

        ],

        constant_values=1

    )

    img = img.reshape(img.shape[0] // p_size, p_size, img.shape[1] // p_size, p_size, 3)

    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, p_size, p_size, 3)

    if len(img) < num_patches:

        img = np.pad(

            img, 

            [

                [0, num_patches - len(img)],

                [0, 0],

                [0, 0],

                [0, 0]

            ],

            constant_values=1

        )

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_patches]

    return np.array(img[idxs])





def glue_to_one_trian(imgs_seq, img_size, num_patches, normalizer=False ):

    side = int(np.sqrt(num_patches))

    img_glue = np.zeros((img_size * side, img_size * side, 3), dtype=np.float32)

    for i, ptch in enumerate(imgs_seq):

        x = i // side

        y = i % side

        img_glue[x * img_size : (x + 1) * img_size, 

                 y * img_size : (y + 1) * img_size, :] = ptch

    

    if normalizer:

        to_transform = staintools.LuminosityStandardizer.standardize((img_glue*255).astype("uint8"))

        img_norm = normalizer.transform(to_transform) 

    else:

        img_norm = None

    return img_norm, img_glue
# Read data

#target = staintools.read_image('../input/staintools/target.png')

img_path = data_dir + '0005f7aaab2800f6170c399693a96917.tiff'

img_patches = get_patches_train(img_path, SEQ_LEN, IMG_SIZE)

_,target= glue_to_one_trian(img_patches, IMG_SIZE, SEQ_LEN)

target = staintools.LuminosityStandardizer.standardize((target*255).astype("uint8"))



# Stain normalize

#normalizer = staintools.StainNormalizer(method='vahadane')

normalizer = ReinhardColorNormalizer()

normalizer.fit(target)

img_path = data_dir + '78fa6eadfc403f3440ef91db24d387b6.tiff'

img_patches = get_patches_train(img_path, SEQ_LEN, IMG_SIZE)

img_norm, img_glue = glue_to_one_trian(img_patches, IMG_SIZE, SEQ_LEN , normalizer)

#fig, axs = plt.subplots(1,3, figsize=(20, 10))

#axs[0].imshow(target)

#axs[1].imshow(img_norm)

#axs[2].imshow(img_glue)

#plt.show
#plt.figure(figsize=(40, 20))

#plt.imshow(img_norm)
%%time

pds = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')

data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

save_dir1 = "/kaggle/train_images/"

if os.path.exists(save_dir1):

    shutil.rmtree(save_dir1)

os.makedirs(save_dir1, exist_ok=True)

with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:

    for im_id in tqdm(pds['image_id'][0:4000]):

        img_path = os.path.join(data_dir,im_id+".tiff")

        if os.path.exists(img_path):

            img_patches = get_patches_train(img_path, SEQ_LEN, IMG_SIZE)

            #img = (glue_to_one_trian(img_patches, IMG_SIZE, SEQ_LEN, normalizer)*255).astype("uint8")

            img, _ = glue_to_one_trian(img_patches, IMG_SIZE, SEQ_LEN , normalizer)

            SIZE_Final1 = int(Scaling * img.shape[0])

            SIZE_Final2 = int(Scaling * img.shape[1])

            img = cv2.resize(img, (SIZE_Final1,SIZE_Final2))

            save_path = save_dir1 + im_id + '.tiff'

            #cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            #Image.fromarray(img).save(im_id+'.jpeg')

            img = cv2.imencode('.jpeg',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]

            img_out.writestr(f'{im_id}.jpeg', img)
#fig, axs = plt.subplots(1,2, figsize=(20, 10))

#im384 = io.imread(os.path.join(save_dir1,os.listdir(save_dir1)[0]))

#axs[0].imshow(im384)

#im980 = io.imread(os.path.join(save_dir2,os.listdir(save_dir2)[0]))

#axs[1].imshow(im980)

#plt.show

#print(im384.shape)

#print(im980.shape)
#!tar -czf train_images.tar.gz ../train_images/*.tiff
#!tar -czf train_images_960.tar.gz ../train_images_960/*.tiff