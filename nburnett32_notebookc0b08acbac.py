# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from os import walk

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import sklearn as sk

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

mypath = "../input/"

f = []

for (dirpath,dirnames,filenames)in walk(mypath):

    f.extend(filenames)

    break



frames = []

n = 0

for myfile in f:

    full_path = mypath+myfile

    print(full_path)

    print ("File: ",myfile)

    

    if n == 0:

        gun_df = pd.read_csv(full_path,header=0)

        n=1

    else:        

        tempdf = pd.read_csv(full_path,header=0)

        gun_df = gun_df.append(tempdf)

    #print(gun_df)



new_df = gun_df.drop_duplicates(keep = 'first')

print(new_df)
d_frame = new_df.sort(columns = 'Incident Date', ascending = True)

print(d_frame)

d_frame['Year'] = Series