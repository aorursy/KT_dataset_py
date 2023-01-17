import numpy as np # linear algebra

import random

import matplotlib.pyplot as plt

import glob

import os

from PIL import Image
# Install prereqs

!git clone https://github.com/dwgoon/jpegio

!pip install jpegio/.

import jpegio as jio
data_dir = '/kaggle/input/alaska2-image-steganalysis/'

base_dir = data_dir + "Cover/"

dirs = {

    'JMiPOD': data_dir + "JMiPOD/",

    'JUNIWARD': data_dir + "JUNIWARD/",

    'UERD': data_dir + "UERD/"

}



nb_imgs = 20

imgList = os.listdir(base_dir)[:nb_imgs]

random.shuffle(imgList)



def get_DCT(im_path):

    c_struct = jio.read(im_path)

    out = np.zeros([512,512,3])

    out[:,:,0] = c_struct.coef_arrays[0]

    out[:,:,1] = c_struct.coef_arrays[1]

    out[:,:,2] = c_struct.coef_arrays[2]

    return out



for im in imgList: 

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    full_path = os.path.join(base_dir, im)

    

    # Print image

    base_pixels = np.array(Image.open(full_path))

    axs[0,0].axis('off')

    axs[0,0].set_title('Original Image')

    axs[0,0].imshow(base_pixels)

    

    # Print |DCT|

    base_DCT = get_DCT(full_path)

    axs[1,0].axis('off')

    axs[1,0].set_title('Original Image |DCT|')

    axs[1,0].imshow(abs(base_DCT))



    for idx, entry in enumerate(dirs.items()):

        name, dir_ = entry

        full_path = os.path.join(dir_, im)

        

        # Print pixel difference

        pixels = np.array(Image.open(full_path))

        pixelsDiff = base_pixels - pixels;

        axs[0,idx+1].axis('off')

        axs[0,idx+1].set_title(f'{name} |$\Delta$Pixel|')

        axs[0,idx+1].imshow(abs(pixelsDiff))

        

        # Print DCT difference

        DCT = get_DCT(full_path)

        imgDiff = base_DCT - DCT;

        axs[1,idx+1].axis('off')

        axs[1,idx+1].set_title(f'{name} |$\Delta$DCT|')

        axs[1,idx+1].imshow(abs(imgDiff))