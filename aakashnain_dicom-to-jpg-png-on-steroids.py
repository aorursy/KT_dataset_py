import warnings

warnings.filterwarnings("ignore")



import os

import cv2

import glob

import time

import pydicom

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pathlib import Path



import dask as dd

import dask.array as da

from dask.distributed import Client, progress



print(os.listdir("../input/sample images"))
# Path to the data

data_dir = Path('../input/sample images/')



# get the list of all the dcm files recursively

all_files = list(data_dir.glob("**/*.dcm"))



print("Number of dcm files found: ", len(all_files))
# Define the path to output directory

outdir = "./processed_images/"



# Make the directory

if not os.path.exists(outdir):

    os.mkdir(outdir)
# Convert DICOM to JPG/PNG via openCV

def convert_images(filename, img_type='jpg'):

    """Reads a dcm file and saves the files as png/jpg

    

    Args:

        filename: path to the dcm file

        img_type: format of the processed file (jpg or png)

        

    """

    # extract the name of the file

    name = filename.parts[-1]

    

    # read the dcm file

    ds = pydicom.read_file(str(filename)) 

    img = ds.pixel_array

    

    # save the image as jpg/png

    if img_type=="jpg":

        cv2.imwrite(outdir + name.replace('.dcm','.jpg'), img)

    else:

        cv2.imwrite(outdir + name.replace('.dcm','.png'), img)
# Making the list bigger hust for showcasing 

all_files = all_files*1000

print("Total number of files: ", len(all_files))
# First using the simple way: the for loop

t = time.time()

for f in all_files:

    convert_images(f)

print("Time taken : ", time.time() - t)
# Using dask 

all_images = [dd.delayed(convert_images)(all_files[x]) for x in range(len(all_files))]



t = time.time()

dd.compute(all_images)

print("Time taken when using all cores: ", time.time()-t)
# Confirm that all the original 10 images are saved 

! ls ./processed_images/* | wc -l