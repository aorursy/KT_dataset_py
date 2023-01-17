setnum = 1
import subprocess

import os

from tqdm import tqdm

import sys

from astropy.io import fits

from astropy.wcs import WCS

import pickle

import numpy as np

import pandas as pd

from tqdm import tqdm
with open("../input/sdss-links-demo/sdss_dictionary_pickle", "rb") as fp:

    obj_dict = pickle.load(fp)
filters = ["u","g","r","i","z"]
obj_list = sorted(obj_dict.keys())

l = len(obj_list)
obj_list = obj_list[l*(setnum - 1)//8 : l*(setnum)//8]
image_dict = {}

for obj in tqdm(obj_list):

    cur_dict = obj_dict[obj]    

    run = cur_dict["run"]

    run6 = "0"*(6 - len(str(run)))+str(run)

    rerun = cur_dict["rerun"]

    camcol = cur_dict["camcol"]

    field = cur_dict["field"]

    field4 = "0"*(4 - len(str(field)))+str(field)

    ra = cur_dict["ra"]

    dec = cur_dict["dec"]

    stacked_arr = []

    for filt in filters:

        link = cur_dict[f"url_{filt}"]

        fname = cur_dict[f"fname_{filt}"]

        command = subprocess.run(['wget',link], stdout=subprocess.PIPE)

        while command.returncode!=0:

            command = subprocess.run(['wget',link], stdout=subprocess.PIPE)

        with fits.open(fname) as hdul:

            hdr = hdul[0].header

            imdata = hdul[0].data

        w = WCS(fname)



        x, y = w.all_world2pix(ra, dec, 1)

        x = int(np.round(x))

        y = int(np.round(y))

        cropped = imdata[y-32:y+32,x-32:x+32]

        stacked_arr.append(cropped)

        os.remove(fname)

        

    final_arr = np.dstack(stacked_arr) #Order is u,g,r,i,z

    final_arr = np.rot90(np.flip(final_arr,0))

    image_dict[obj] = final_arr
import pickle

with open(f"image_dict_pickleset_{setnum}","wb") as fp:

    pickle.dump(image_dict,fp)
import matplotlib.pyplot as plt

from scipy import ndimage

plt.imshow(np.flip(ndimage.gaussian_filter(image_dict[obj][:,:,2:5],1.25),2))