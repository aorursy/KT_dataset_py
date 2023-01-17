import sys

import os

import subprocess

from six import string_types



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import scipy

from skimage import io

from scipy import ndimage

import rasterio

import matplotlib

import PIL

from IPython.display import display

%matplotlib inline



import os

import openslide
# Localização das imagens de treinamento

data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'

mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks'

ROOT = "/kaggle/input/prostate-cancer-grade-assessment/"

!ls {ROOT}
df_train = pd.read_csv(ROOT + 'train.csv')# .set_index('image_id')

df_test  = pd.read_csv(ROOT + 'test.csv' )#.set_index('image_id')

df_sub   = pd.read_csv(ROOT+"sample_submission.csv")
df_train.head()
df_train.info()
ax = sns.countplot(x="isup_grade", data=df_train)

plt.title("Distrituição Targent")

plt.show()
ax = sns.countplot(x="data_provider", data=df_train)

plt.title("Fornecedor de Dados")

plt.show()
ax = sns.countplot(x="isup_grade", hue="data_provider", data=df_train)

plt.show()
ax = sns.countplot(x="data_provider", data=df_train)

plt.title("Distribuição dados de Treinamento")

plt.show()

ax = sns.countplot(x="data_provider", data=df_test)

plt.title("Distribuição dados de test")

plt.show()
fig= plt.figure(figsize=(10,6))

ax = sns.countplot(y="gleason_score", data=df_train)

plt.tight_layout()

plt.show()



fig= plt.figure(figsize=(10,6))

ax = sns.countplot(y="gleason_score",hue='data_provider', data=df_train)

plt.tight_layout()

plt.show()
def display_images(slides):

    

    for slide in slides:

        image   = openslide.OpenSlide(os.path.join(data_dir, f'{slide}.tiff'))

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch   = image.read_region((1780,1950), 0, (256, 256))

        display(patch) # Display the image

        

        print(f"File id: {slide}")

        print(f"Dimensions: {image.dimensions}")

        print(f"Microns per pixel / pixel spacing: {spacing:.3f}")

        print(f"Number of levels in the image: {image.level_count}")

        print(f"Downsample factor per level: {image.level_downsamples}")

        print(f"Dimensions of levels: {image.level_dimensions}")

        

        # Print the case-level label

        #print(f"ISUP grade: {df_train.loc[slide, 'isup_grade']}")

        #print(f"Gleason score: {df_train.loc[slide, 'gleason_score']}\n\n")

        

        image.close() 
images = [

    '00a76bfbec239fd9f465d6581806ff42',

    '037504061b9fba71ef6e24c48c6df44d',

    '035b1edd3d1aeeffc77ce5d248a01a53',

    '059cbf902c5e42972587c8d17d49efed',

    '06a0cbd8fd6320ef1aa6f19342af2e68',

    '06eda4a6faca84e84a781fee2d5f47e1',

    '0a4b7a7499ed55c71033cefb0765e93d',

    '0ac4677348cf4fc1ebe354ba1037921c',

]





display_images(images)
def display_mask_images(slides, center='radboud', show_thumbnail=True, max_size=(400,400)):

    """Imprimir algumas informações básicas sobre um slide"""

    

    f, ax = plt.subplots(4, 2, figsize=(16, 16))

    

    ax = ax.flatten()

    

    for index, slide in enumerate(slides):

        

        # Gere uma miniatura de imagem pequena

        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))

        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

        

        # Opcional: crie um mapa de cores personalizado

        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

        ax[index].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)

        ax[index].set_xlabel(f'{slide}_mask.tiff')

        f.tight_layout()
display_mask_images(images)
def overlay_mask_on_slide(images, center='radboud', alpha=0.8, max_size=(800, 800)):

    """Show a mask overlayed on a slide."""



    

    for image in images:

        slide = openslide.OpenSlide(os.path.join(data_dir, f'{image}.tiff'))

        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{image}_mask.tiff'))

        # Load data from the highest level

        

        slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])

        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])



        # Mask data is present in the R channel

        mask_data = mask_data.split()[0]



        # Create alpha mask

        alpha_int = int(round(255*alpha))

        if center == 'radboud':

            alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)

        elif center == 'karolinska':

            alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)



        alpha_content = PIL.Image.fromarray(alpha_content)

        preview_palette = np.zeros(shape=768, dtype=int)



        if center == 'radboud':

            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}

            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)

        elif center == 'karolinska':

            # Mapping: {0: background, 1: benign, 2: cancer}

            preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)



        mask_data.putpalette(data=preview_palette.tolist())

        mask_rgb = mask_data.convert(mode='RGB')



        overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)

        overlayed_image.thumbnail(size=max_size, resample=0)



        display(overlayed_image)

        

        print(f"File id: {image}")

        # Print the case-level label

        #print(f"ISUP grade: {df_train.loc[image, 'isup_grade']}")

        #print(f"Gleason score: {df_train.loc[image, 'gleason_score']}\n\n")

        

        

        slide.close()

        mask.close()
overlay_mask_on_slide(images)