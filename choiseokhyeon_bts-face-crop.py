# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io

from PIL import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



#os.mkdir('/kaggle/working/local_crop_img/')

cur_list = open('/kaggle/input/bts-face-local-list/local_list.txt', 'r')



while True:

    line = cur_list.readline()

    if not line: break

    path_save = False

    i=0

    k=0

    cut_=[]

    #print(line)

    while i < len(line):

        if(line[i]==','):

            if path_save:

                cut_.append(int(line[k:i])) 

            else:

                path=line[0:i]

                path_save = True

            k=i+1

        i=i+1

    cut_.append(int(line[k:-1]))

    

    cut_path = '/kaggle/working/local_crop_img/'+path[-9:]

    area=[]

    area.append(cut_[3])

    area.append(cut_[0])

    area.append(cut_[1])

    area.append(cut_[2])

    

    #print(area)

    

    img= Image.open(path)

    crop_img=img.crop(area)

    #crop_img=crop_img.resize( __to 2D input size__ )

    crop_img.save(cut_path)

    



# Any results you write to the current directory are saved as output.