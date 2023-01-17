# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

from scipy.misc import imread



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
sub_folders = check_output(["ls", "../input/train/"]).decode("utf8").strip().split('\n')

count_dict = {}

for sub_folder in sub_folders:

    num_of_files = len(check_output(["ls", "../input/train/"+sub_folder]).decode("utf8").strip().split('\n'))

    print("{0} photos of cervix type {1} ".format(num_of_files, sub_folder))



    count_dict[sub_folder] = num_of_files

    

plt.figure(figsize=(12,4))

sns.barplot(list(count_dict.keys()), list(count_dict.values()), alpha=0.8)

plt.xlabel('Cervix types', fontsize=12)

plt.ylabel('Number of Images', fontsize=12)

plt.show()
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

plt.xlabel('Number of Images', fontsize=12)

plt.title("Image sizes present in train dataset")

plt.show()
train_path = "../input/train/"

sub_folders = check_output(["ls", train_path]).decode("utf8").strip().split('\n')

different_file_sizes = {}

for sub_folder in sub_folders:

    file_names = check_output(["ls", train_path+sub_folder]).decode("utf8").strip().split('\n')

    for file_name in file_names:

        img_path = train_path+sub_folder+"/"+file_name

        with Image.open(img_path) as img:

            (width, heigh) = img.size

            if width < heigh:

                size = str(heigh) + "x" + str(width)

            else:

                size = str(width) + "x" + str(heigh)

            different_file_sizes[size] = different_file_sizes.get(size,0) + 1



plt.figure(figsize=(12,4))

sns.barplot(list(different_file_sizes.values()), list(different_file_sizes.keys()), alpha=0.8)

plt.ylabel('Image size', fontsize=12)

plt.xlabel('Number of Images', fontsize=12)

plt.title("Image sizes present in train dataset")

plt.show()
test_path = "../input/test/"

sub_folders = check_output(["ls", test_path]).decode("utf8").strip().split('\n')

different_file_sizes = {}

file_names = check_output(["ls", test_path]).decode("utf8").strip().split('\n')

for file_name in file_names:

    img_path = test_path+"/"+file_name

    with Image.open(img_path) as img:

        (width, heigh) = img.size

        if width < heigh:

            size = str(heigh) + "x" + str(width)

        else:

            size = str(width) + "x" + str(heigh)

        different_file_sizes[size] = different_file_sizes.get(size,0) + 1



plt.figure(figsize=(12,4))

sns.barplot(list(different_file_sizes.values()), list(different_file_sizes.keys()), alpha=0.8)

plt.ylabel('Image size', fontsize=12)

plt.xlabel('Number of Images', fontsize=12)

plt.title("Image sizes present in train dataset")

plt.show()