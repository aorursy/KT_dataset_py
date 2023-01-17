# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

#from scipy.misc import imread

from glob import glob

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/train/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#fig = plt.figure(figsize=(12,8))

size = 256, 256



im=Image.open('../input/train/train/Type_1/10.jpg')

im.thumbnail(size, Image.ANTIALIAS)

#print (im.format, im.size, im.mode)

im=Image.open('../input/train/train/Type_2/100.jpg')

im.thumbnail(size, Image.ANTIALIAS)

#print (im.format, im.size, im.mode)

im=Image.open('../input/train/train/Type_3/1000.jpg')

im.thumbnail(size, Image.ANTIALIAS)

#print (im.format, im.size, im.mode)

sub_folders = check_output(["ls", "../input/train/"]).decode("utf8").strip().split('\n')

count_dict = {}

for sub_folder in sub_folders:

    num_of_files = len(check_output(["ls", "../input/train/"+sub_folder]).decode("utf8").strip().split('\n'))

    print("{0} photos of cervix type {1} ".format(num_of_files, sub_folder))



    count_dict[sub_folder] = num_of_files

    

plt.figure(figsize=(12,4))

sns.barplot(list(count_dict.keys()), list(count_dict.values()), alpha=0.8)

plt.xlabel('Cervix types', fontsize=12)

plt.ylabel('Number of Images in train', fontsize=12)

plt.title("train dataset")



plt.show()

    
sub_folders = check_output(["ls", "../input/additional/"]).decode("utf8").strip().split('\n')

count_dict = {}

for sub_folder in sub_folders:

    num_of_files = len(check_output(["ls", "../input/additional/"+sub_folder]).decode("utf8").strip().split('\n'))

    print("{0} photos of cervix type {1} ".format(num_of_files, sub_folder))



    count_dict[sub_folder] = num_of_files

    

plt.figure(figsize=(12,4))

sns.barplot(list(count_dict.keys()), list(count_dict.values()), alpha=0.8)

plt.xlabel('Cervix types', fontsize=12)

plt.ylabel('Number of Images in additional', fontsize=12)

plt.title("Additional dataset")

plt.show()

    
num_test_files = len(check_output(["ls", "../input/test/"]).decode("utf8").strip().split('\n'))

print("Number of test images present :", num_test_files)
train_path = "../input/train/"

sub_folders = check_output(["ls", train_path]).decode("utf8").strip().split('\n')

different_file_sizes = {}

for sub_folder in sub_folders:

    file_names = check_output(["ls", train_path+sub_folder]).decode("utf8").strip().split('\n')

    for file_name in file_names:

        im_array = imread(train_path+sub_folder+"/"+file_name)

        size = "_".join(map(str,list(im_array.shape)))

        different_file_sizes[size] = different_file_sizes.get(size,0) + 1



plt.figure(figsize=(12,4))

sns.barplot(list(different_file_sizes.values()), list(different_file_sizes.keys()), alpha=0.8)

plt.ylabel('Image size', fontsize=12)

plt.xlabel('Number of Images in train', fontsize=12)

plt.title("Image sizes present in train dataset")

plt.show()
test_path = "../input/test/test/"

file_names = check_output(["ls", test_path]).decode("utf8").strip().split('\n')

different_file_sizes = {}

for file_name in file_names:

        size = "_".join(map(str,list(Image.open(test_path+file_name).size)))

        different_file_sizes[size] = different_file_sizes.get(size,0) + 1



plt.figure(figsize=(12,4))

sns.barplot(list(different_file_sizes.keys()), list(different_file_sizes.values()), alpha=0.8)

plt.xlabel('File size', fontsize=12)

plt.ylabel('Number of Images in test', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Image size present in test dataset")

plt.show()
additional_path = "../input/additional_Type_1_v2/"

sub_folders = check_output(["ls", additional_path]).decode("utf8").strip().split('\n')

different_file_sizes = {}

#corrupted_images=['2845.jpg','5892.jpg','5893.jpg']

for sub_folder in sub_folders:

    file_names = check_output(["ls", additional_path+sub_folder]).decode("utf8").strip().split('\n')

    #if(sub_folder=='Type_2'):

     #   try:

      #      file_names.remove('5892.jpg')

       #     file_names.remove('5893.jpg')

        #    file_names.remove('2845.jpg')

        #except ValueError:

         #   pass

    for file_name in file_names:

        im_array = Image.open(additional_path+sub_folder+"/"+file_name)

        size = "_".join(map(str,list(im_array.size)))

        different_file_sizes[size] = different_file_sizes.get(size,0) + 1



plt.figure(figsize=(12,4))

sns.barplot(list(different_file_sizes.values()), list(different_file_sizes.keys()), alpha=0.8)

plt.ylabel('Image size', fontsize=12)

plt.xlabel('Number of Images in additional', fontsize=12)

plt.title("Image sizes present in additional dataset")

plt.show()