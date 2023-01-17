import pandas as pd

from PIL import Image

import numpy as np

import tqdm 

import os

import glob

import openslide

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

import cv2

import skimage.io

import PIL

from skimage.transform import resize, rescale
input_path = "../input/prostate-cancer-grade-assessment"
listi = []

train_images_path = os.path.join(input_path,"train_images")

images = glob.glob(os.path.join(train_images_path, "*tiff"))

image_label = pd.read_csv(input_path + "/train.csv")["image_id"]

for i in tqdm.tqdm(range(len(images))):

    image = openslide.OpenSlide(images[i])   

    listi.append((image_label[i],image.dimensions))
width = [i[1][0] for i in listi]

plt.hist(width)
height = [i[1][1] for i in listi]

plt.hist(height)
mask_images_path = os.path.join(input_path,"train_label_masks")

masked_images = os.listdir(mask_images_path)
def overlay_mask_on_slide(image_id, center='radboud', alpha=0.8, max_size=(800, 800)):

    """Show a mask overlayed on a slide."""

    slide = openslide.OpenSlide(os.path.join(train_images_path, f'{image_id}.tiff'))

    mask = openslide.OpenSlide(os.path.join(mask_images_path, f'{image_id}_mask.tiff'))

    slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

    mask_data = mask_data.split()[0]

      

        

    # Create alpha mask

    alpha_int = int(round(255*alpha))

    if center == 'radboud':

        alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)

    elif center == 'karolinska':

        alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)



    alpha_content = Image.fromarray(alpha_content)

    preview_palette = np.zeros(shape=768, dtype=int)



    if center == 'radboud':

        # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

        preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

    elif center == 'karolinska':

        # Mapping: {0: background, 1: benign, 2: cancer}

        preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)



    mask_data.putpalette(data=preview_palette.tolist())

    mask_rgb = mask_data.convert(mode='RGB')

    overlayed_image = Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)

    overlayed_image.thumbnail(size=max_size, resample=0)

    

    slide.close()

    mask.close()   

          

    return overlayed_image
!mkdir train_overlay_images
train = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')



def image_masking_fix(img_id):

    img_id = img_id.replace('_mask.tiff', '')

    save_path = save_dir + img_id + '.png'

    provider = train[train.image_id == img_id]['data_provider'].values[0]

    overlay = overlay_mask_on_slide(img_id, center = provider)

    img = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (512, 512))

    cv2.imwrite(save_path, img)
save_dir = "train_overlay_images/" 

os.makedirs(save_dir, exist_ok=True)



Parallel(n_jobs=32, verbose=10)(delayed(image_masking_fix)(f) for f in tqdm.tqdm(masked_images))