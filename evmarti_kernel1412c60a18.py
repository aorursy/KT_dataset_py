# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



img_folder_path = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized'

para = os.listdir(img_folder_path)



print(len(para))

print("there are 13780 parasitized cells in the database")





img_folder_path2 = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected'

uninfected = os.listdir(img_folder_path)



print(len(uninfected))  

print('there are 13780 uninfected cell files in the database')

#Hi Professor Koffi!

    

    

    

    

# Any results you write to the current directory are saved as output.