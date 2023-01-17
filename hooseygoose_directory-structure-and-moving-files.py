import numpy as np

import os

import pandas as pd

import shutil



from shutil import unpack_archive

from subprocess import check_output
path = "./Leaf Classification/" # specify directory where the competition zip files were downloaded

fileList = os.listdir(path) # get file list in the path directory

# list files

for f in fileList: 

    print(f)
# unpack zip files in path directory

for i in fileList:

    if "zip" in i:

        unpack_archive(path + i, path, 'zip')
fileList = os.listdir(path) # get updated file list in path directory

# list files/folders

for f in fileList:

    print(f)
# populate train.csv into a pandas DataFrame

train = pd.read_csv(path + "train.csv")

print(train.shape) # print DataFrame shape

train.head() # print first 5 records of the DataFrame
# populate test.csv into a pandas DataFrame

test = pd.read_csv(path + "test.csv")

print(test.shape) # print DataFrame shape

test.head() # print first 5 records of the DataFrame
# create DataFrame containing a sorted list of species from the train DataFrame

leafClasses = train.species.sort_values(axis=0)

# create Dataframe containing a unique list of specieas from the train DataFrame

leafClassesUnique = leafClasses.unique()
# create train, valid, and test directories

shutil.os.mkdir(path + "train")

shutil.os.mkdir(path + "valid")

shutil.os.mkdir(path + "test")
# create directories for each unique species in the leafClassesUnique Dataframe in the train and valid directories previous created

for i in leafClassesUnique:

    shutil.os.mkdir(path + "train/" + i)

    shutil.os.mkdir(path + "valid/" + i)
# move the test images from the images directory to the test directory

for i in test.iloc[:,0]: # test image labels are identified in the id column of the test DataFrame

    shutil.move(path + "images/"+ str(i) + ".jpg", path + "test")
# move the training images from the images directory to the train directory

# since the test images are no longer in the folder, it is not necessary to identify every image sperately.

# the images were specifically identified to determine any images were not associated with the train or test DataFrames

for index, row in train.iterrows():

    shutil.move(path + "images/" + str(row[0]) +".jpg",path + "train/" + row[1])
# create validation set from the train set. 

for i in leafClassesUnique: # loop through the species list which alligns with the structure of the train and valid directories

    # populate DateFrame with file names in the specific species folder in the train directory

    df = os.listdir(path + "train/" + i) 

    # each species folder in the train directory contains only 10 images. 3 of the 10 images were chosen for the validation set

    shutil.move(path + "train/" + i + "/" + df[1],path + "valid/" + i)

    shutil.move(path + "train/" + i + "/" + df[3],path + "valid/" + i)

    shutil.move(path + "train/" + i + "/" + df[7],path + "valid/" + i)