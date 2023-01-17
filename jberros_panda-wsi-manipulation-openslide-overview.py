%matplotlib inline

import os



import numpy as np

import openslide

from openslide import deepzoom

from matplotlib import pyplot as plt
#Images / Masks Directories

images_dir = "../input/prostate-cancer-grade-assessment/train_images/"

masks_dir = "../input/prostate-cancer-grade-assessment/train_label_masks/"



#Files

image_files = os.listdir(images_dir)

mask_files = os.listdir(masks_dir)

mask_files_cleaned = [i.replace("_mask", "") for i in mask_files]



#Clean Images without Masks

images_with_masks = list(set(image_files).intersection(mask_files_cleaned))

len(image_files), len(mask_files), len(images_with_masks)
image_file = '00928370e2dfeb8a507667ef1d4efcbb.tiff'

mask_file = '00928370e2dfeb8a507667ef1d4efcbb_mask.tiff'
image = openslide.OpenSlide(os.path.join(images_dir, image_file))

mask = openslide.OpenSlide(os.path.join(masks_dir, mask_file))
print(type(image))

type(mask)
img_level_count = image.level_count

mask_level_count = mask.level_count

print(img_level_count, mask_level_count)
detect_format = openslide.PROPERTY_NAME_VENDOR

print(detect_format)
image_size1 = image.level_dimensions[0]

image_size2 = image.level_dimensions[1]

image_size3 = image.level_dimensions[2]

print(image_size1, image_size2, image_size3)
mask_size1 = mask.level_dimensions[0]

mask_size2 = mask.level_dimensions[1]

mask_size3 = mask.level_dimensions[2]

print(mask_size1, mask_size2, mask_size3)
image_dwn1 = image.level_downsamples[0]

image_dwn2 = image.level_downsamples[1]

image_dwn3 = image.level_downsamples[2]

print(image_dwn1, image_dwn2, image_dwn3)
mask_dwn1 = mask.level_downsamples[0]

mask_dwn2 = mask.level_downsamples[1]

mask_dwn3 = mask.level_downsamples[2]

print(mask_dwn1, mask_dwn2, mask_dwn3)
image.properties
mask.properties
image.associated_images
mask.associated_images
image.read_region((0, 0), 2 , (512, 512))
mask.read_region((0, 0), 2 , (512, 512))
image.get_best_level_for_downsample(18)
image.get_thumbnail((300,300))
zoom = openslide.deepzoom.DeepZoomGenerator(image, tile_size=254, overlap=1, limit_bounds=False)
print(type(zoom))
zoom.level_count
zoom.tile_count
zoom.level_tiles
zoom.level_dimensions
zoom.get_dzi('png')
format(zoom)
zoom.get_tile(13, (2, 5))
zoom.get_tile_coordinates(13, (2, 5))
zoom.get_tile_dimensions(13, (2, 5))