import SimpleITK as sitk

import numpy as np

import csv

from glob import glob

import pandas as pd

file_list=glob(luna_subset_path+"*.mhd")

#####################

#

# Helper function to get rows in data frame associated 

# with each file

def get_filename(case):

    global file_list

    for f in file_list:

        if case in f:

            return(f)

#

# The locations of the nodes

df_node = pd.read_csv(luna_path+"annotations.csv")

df_node["file"] = df_node["seriesuid"].apply(get_filename)

df_node = df_node.dropna()

#####

#

# Looping over the image files

#

fcount = 0

for img_file in file_list:

    print ("Getting mask for image file %s" % img_file.replace(luna_subset_path,""))

    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file

    if len(mini_df)>0:       # some files may not have a nodule--skipping those 

        biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node

        node_x = mini_df["coordX"].values[biggest_node]

        node_y = mini_df["coordY"].values[biggest_node]

        node_z = mini_df["coordZ"].values[biggest_node]

        diam = mini_df["diameter_mm"].values[biggest_node]
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from skimage import morphology

from skimage import measure

from sklearn.cluster import KMeans

from skimage.transform import resize

print ('ok')