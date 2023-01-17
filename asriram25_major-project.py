# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *

import os

from os import listdir

%reload_ext autoreload

%autoreload 2

%matplotlib inline

path = "../input/rice-leaf-diseases/"

os.listdir(path)
directory_root = '../input/rice-leaf-diseases/'

image_list, label_list = [], []

try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root)

    for directory in root_dir :

        # remove .DS_Store from list

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    for plant_disease_folder in root_dir:

        print(f"[INFO] Processing {plant_disease_folder} ...")

        plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")

                

        for single_plant_disease_image in plant_disease_image_list :

            if single_plant_disease_image == ".DS_Store" :

                plant_disease_image_list.remove(single_plant_disease_image)



        for image in plant_disease_image_list[:200]:

            image_directory = f"{directory_root}/{plant_disease_folder}/{image}"

            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                image_list.append(image_directory)

                label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
tfms = get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=0.)

def get_labels(file_path): 

    dir_name = os.path.dirname(file_path)

    split_dir_name = dir_name.split("/")

    dir_length = len(split_dir_name)

    label  = split_dir_name[dir_length - 1]

    return(label)

data = ImageDataBunch.from_name_func(path, image_list, label_func=get_labels,  size=224, 

                                     bs=64,num_workers=2,ds_tfms=tfms)

data = data.normalize()
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/tmp/models/')
learn.fit_one_cycle(15)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.recorder.plot_losses()
ls
learn.save('model')

learn.export(file = Path("/kaggle/working/export.pkl"))
from IPython.display import FileLinks

FileLinks('.')
