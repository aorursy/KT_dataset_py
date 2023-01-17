import fastai.vision

import os

import glob

from multiprocessing import Pool

from functools import partial

import matplotlib.pyplot as plt

import random

import numpy as np

from tqdm import tqdm_notebook as tqdm

import pandas as pd

from zipfile import ZipFile 
xray_images= fastai.vision.ImageList.from_folder('../input/siimdatasetx128/siim-datasetx128/siim-datasetx128/train/128/dicom/')

mask_images = fastai.vision.ImageList.from_folder('../input/siimdatasetx128/siim-datasetx128/siim-datasetx128/train/128/mask/')

print(len(xray_images),len(mask_images))
def plot_image_mask(img_,mask_):

    _,axs = plt.subplots(1,3, figsize=(15,7))

    img_.show(ax=axs[0], title='no mask')

    img_.show(ax=axs[1], y=mask_, title='masked')

    mask_.show(ax=axs[2], title='mask only', alpha=1.)

    

img_exemple = fastai.vision.open_image(xray_images.items[2974])

mask_exemple = fastai.vision.open_mask(mask_images.items[2974])



plot_image_mask(img_exemple,mask_exemple)
_ROTATE_DEGREES = -50

_MAGNITUDE_WRAP = -0.2

_SCALE_ZOOM = 1.5

_CHANGE_BRIGHTNESS = 0.8

_SCALE_CONSTRAST = (2,3)

tfms_dicom=[

    fastai.vision.rotate(degrees = _ROTATE_DEGREES),

    fastai.vision.symmetric_warp(magnitude = _MAGNITUDE_WRAP),

    fastai.vision.zoom(scale = _SCALE_ZOOM),

    fastai.vision.brightness(change = _CHANGE_BRIGHTNESS),

    fastai.vision.contrast(scale = _SCALE_CONSTRAST)]



tfms_mask=[

    fastai.vision.rotate(degrees = _ROTATE_DEGREES),

    fastai.vision.symmetric_warp(magnitude = _MAGNITUDE_WRAP),

    fastai.vision.zoom(scale = _SCALE_ZOOM)]



img_exemple_generated = img_exemple.apply_tfms(tfms_dicom,padding_mode="zeros")

mask_exemple_generated = mask_exemple.apply_tfms(tfms_mask,padding_mode="zeros")



plot_image_mask(img_exemple_generated,mask_exemple_generated)

img = fastai.vision.open_image(xray_images.items[10620])

mask = fastai.vision.open_mask(mask_images.items[10620])

plot_image_mask(img,mask)
fastai.vision.rle_encode(mask.data)
img = fastai.vision.open_image(xray_images.items[6])

mask = fastai.vision.open_mask(mask_images.items[6])

plot_image_mask(img,mask)
def generate_new_images(img_,mask_):

    #select random paramters to apply

    _ROTATE_DEGREES  = random.choices(np.linspace(-60,60,12))[0]

    _MAGNITUDE_WRAP = random.choices(np.linspace(-0.2,-0.1,3))[0]

    _SCALE_ZOOM = random.choices(np.linspace(1.1,1.5,6))[0]

    _CHANGE_BRIGHTNESS = random.choices(np.linspace(0.6,0.8,10))[0]

    _SCALE_CONSTRAST = random.choices([(i,j) for i,j in zip(np.linspace(2,4,1),np.linspace(4,7,1))])[0]

    tfms_dicom=[

        fastai.vision.rotate(degrees = _ROTATE_DEGREES),

        fastai.vision.symmetric_warp(magnitude = _MAGNITUDE_WRAP),

        fastai.vision.zoom(scale = _SCALE_ZOOM),

        fastai.vision.brightness(change = _CHANGE_BRIGHTNESS),

        fastai.vision.contrast(scale = _SCALE_CONSTRAST)]



    tfms_mask=[

        fastai.vision.rotate(degrees = _ROTATE_DEGREES),

        fastai.vision.symmetric_warp(magnitude = _MAGNITUDE_WRAP),

        fastai.vision.zoom(scale = _SCALE_ZOOM)]



    img_generated = img_.apply_tfms(tfms_dicom,padding_mode="zeros")

    mask_generated = mask_.apply_tfms(tfms_mask,padding_mode="zeros")

    #define an ID to save in csv file

    suffix_imageid = f'_generated_{int(np.abs(_ROTATE_DEGREES))}_{int(np.abs(_MAGNITUDE_WRAP))}_{int(_SCALE_ZOOM)}_{int(_CHANGE_BRIGHTNESS)}_{int(_SCALE_CONSTRAST[0])}_{int(_SCALE_CONSTRAST[1])}'

    return img_generated,mask_generated, suffix_imageid



img_generated,mask_generated, suffix_imageid = generate_new_images(img,mask)



plot_image_mask(img_generated,mask_generated)

print('MASK GENERATED: {}'.format(fastai.vision.rle_encode(mask_generated.data)))
def generate_new_data(inputdir_dicom,inputdir_mask,outputdir,number_of_data_to_generate):

    xray_images= fastai.vision.ImageList.from_folder(inputdir_dicom)

    mask_images = fastai.vision.ImageList.from_folder(inputdir_mask)

    print(len(xray_images),len(mask_images))

    outputdir_dicom = outputdir+'dicom/'

    outputdir_mask = outputdir+'mask/'

    

    if not os.path.exists(outputdir_dicom):

        os.makedirs(outputdir_dicom)

    if not os.path.exists(outputdir_mask):

        os.makedirs(outputdir_mask)

    

    ids_generated = []

    rle_generated = []

    number_of_data_to_generate_=number_of_data_to_generate

    for i in tqdm(range(0,number_of_data_to_generate_)):

        random_choice = random.randint(0,len(xray_images))

        img = fastai.vision.open_image(xray_images.items[random_choice])

        mask = fastai.vision.open_mask(mask_images.items[random_choice])

        img_generated,mask_generated, suffix_imageid = generate_new_images(img,mask)

        

        #plot_image_mask(img_generated,mask_generated)

        

        ImageId = os.path.splitext(os.path.basename(xray_images.items[random_choice]))[0]

        #print(outputdir_dicom+ImageId+suffix_imageid+'.png')

        #print(outputdir_mask+ImageId+suffix_imageid+'.png')

        try:

            rle_encoded = fastai.vision.rle_encode(mask_generated.data)

        except ValueError:

            number_of_data_to_generate= number_of_data_to_generate-1 # @TODO: Need to fix this error

            continue

        if rle_encoded == '':

            rle_encoded = ' -1'

        ids_generated.append(ImageId+suffix_imageid)

        img_generated.save(outputdir_dicom+ImageId+suffix_imageid+'.png')

        mask_generated.save(outputdir_mask+ImageId+suffix_imageid+'.png')

        rle_generated.append(rle_encoded)

    print('END : {} IMAGE GENERATED'.format(number_of_data_to_generate))

    df_rle_generated = pd.DataFrame({'ImageId':ids_generated,' EncodedPixels':rle_generated},columns=['ImageId',' EncodedPixels'])

    df_rle_generated.to_csv('./train-rle_generated.csv',index=False, header=True)

    print('RLE CSV DATA SAVED')



inputdir_dicom = '../input/siimdatasetx128/siim-datasetx128/siim-datasetx128/train/128/dicom/'

inputdir_mask = '../input/siimdatasetx128/siim-datasetx128/siim-datasetx128/train/128/mask/'

path_to_save_generated_data = '../output_data_generation/'



generate_new_data(inputdir_dicom,inputdir_mask,path_to_save_generated_data,1000)
def zip_image_generated(path_to_images_folder):

    images_to_compress = fastai.vision.ImageList.from_folder(path_to_images_folder)

    if len(images_to_compress) > 1:

        print('---------------------------')

        print('{0} Image to compress in {1}'.format(len(images_to_compress),path_to_images_folder))

        with ZipFile('./images_generated/image_generated.zip','w') as zip: 

        # writing each file one by one 

            for i in range(0,len(images_to_compress)):

                zip.write(images_to_compress.items[i]) 

        print('{} Image compressed'.format(len(images_to_compress)))

        print('---------------------------')

    else:

        print('No images in {}, please check again'.format(path_to_images_folder))

        raise ValueError



zip_image_generated('../output_data_generation/images_generated/dicom/')

zip_image_generated('../output_data_generation/images_generated/mask/')
pd.read_csv('./train-rle_generated.csv')