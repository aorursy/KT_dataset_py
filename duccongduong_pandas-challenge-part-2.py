# Imports

import os



import openslide

from IPython.display import Image, display

#     Allows viewing of images



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Setting dataset directories to variables



train_dir = "/kaggle/input/prostate-cancer-grade-assessment/train_images/"

#     train_images



mask_dir = "/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/"

#     train_maskes
# Viewing train images using OpenSlide and IPython.display

# Special Thanks to the getting started guide: https://www.kaggle.com/wouterbulten/getting-started-with-the-panda-dataset



image_name = "00412139e6b04d1e1cee8421f38f6e90.tiff"

#     file_name



slide = openslide.OpenSlide(os.path.join(train_dir, image_name))

#     Creates OpenSlide object at the directed path variable

#     os.path.join creates a path string inserting '/' if needed



display(slide.get_thumbnail(size=(400,500)))

#     Opens the image and displays a thumbnail in the notebook



slide.close()

#     Close image from memory
# We can also view a subsection of the image if you want

image_name = "005e66f06bce9c2e49142536caf2f6ee.tiff"

slide = openslide.OpenSlide(os.path.join(train_dir, image_name))

display(slide.read_region((17800,19500), 0, (500,500)))

#     At location X = 17800 and Y = 19500

#     At image level zero 0

#     View a 500 by 500 pixel region of the image

slide.close()

# We can also view an images properties

image_name = "00412139e6b04d1e1cee8421f38f6e90.tiff"

slide = openslide.OpenSlide(os.path.join(train_dir, image_name))

properties = str(slide.properties)

#     Print commands always occur at the end of the cell so the picture properties must be grabbed before closing the slide.

slide.close()



print(properties)

# Better View of the Properties

import ast



ast.literal_eval(properties.strip("<,>,_,PropertyMap,").strip())

#     literal_eval can convert a string dictionary into a dictionary for an easier print format

#     But we first have to remove extra characters and extra space 
# Viewing Image Masks using MatPlotLib

image_name = "00412139e6b04d1e1cee8421f38f6e90_mask.tiff"

slide = openslide.OpenSlide(os.path.join(mask_dir, image_name))

plt.imshow(slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1]).split()[0])

#     plt.imshow let's you show a matrix of numbers as an image, which is what a mask is

#     read_region still works on this matrix, but it spits out four different matrixs 

#     Only the first matrix in the list has the relevant information for plotting

slide.close()
from os import listdir

#     listdir generates a list of items in the directory
# Create a list of the three for list comparisions



list_trainimages = listdir(train_dir)

list_trainmask = listdir(mask_dir)

list_imageid = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')['image_id']
print(f"""

Train Images: {len(list_trainimages)}

Train Masks: {len(list_trainmask)}

Image Ids: {len(list_imageid)}

""")
# Let's find the images that don't have masks



list_trainimages = [image.split(".")[0] for image in list_trainimages]

list_trainmask = [mask.split("_")[0] for mask in list_trainmask]

#     Removes .tiff and _mask.tiff from each file name
for image in list_trainimages:

    if image not in list_trainmask:

        print(image)