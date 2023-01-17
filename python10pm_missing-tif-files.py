import numpy as np

import pandas as pd



import os



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image
total_l = []

tfw_l = []

tif_l = []



for dirname, _, filenames in os.walk('../input/lyft-motion-prediction-autonomous-vehicles/aerial_map/nearmap_images/'):

    for filename in filenames:

        extension = os.path.splitext(filename)[1]

        

        total_l.append(filename)

        

        if extension in ".tfw":

            tfw_l.append(os.path.splitext(filename)[0])

            

        elif extension in ".tif":

            tif_l.append(os.path.splitext(filename)[0])
print(len(tfw_l) == len(tif_l),len(total_l), len(tfw_l), len(tif_l))
intersect = set(tfw_l).intersection(set(tif_l))
len(intersect)
RANDOM_CHOICE = np.random.choice(list(intersect))



def plot_tif(RANDOM_CHOICE):

    try:

        

        PATH_TFW = "../input/lyft-motion-prediction-autonomous-vehicles/aerial_map/nearmap_images/{}.tfw".format(RANDOM_CHOICE)

        PATH_TIF = "../input/lyft-motion-prediction-autonomous-vehicles/aerial_map/nearmap_images/{}.tif".format(RANDOM_CHOICE)



        file_tfw = open(PATH_TFW, "r").read()

        print("Data contained in the {} file is {}.".format(RANDOM_CHOICE, file_tfw.split("\n")))



        fig = plt.figure(figsize = (10, 10))

        ax = fig.add_subplot()



        img = Image.open(PATH_TIF)

        new = img.resize((1024, 1024), resample = Image.BILINEAR)



        ax.imshow(new)

        

    except Exception as e:

        print(e)

        

plot_tif(RANDOM_CHOICE)
no_intersect = set(tfw_l).symmetric_difference(set(tif_l))



RANDOM_CHOICE = np.random.choice(list(no_intersect))



plot_tif(RANDOM_CHOICE)