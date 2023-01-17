    # References:https://www.kaggle.com/c/data-science-bowl-2017/kernels

import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

        for filename in filenames:

            print(os.path.join(dirname, filename))

%matplotlib inline

import pydicom

main_dir ='/kaggle/input/'

patients_data = os.listdir(main_dir)



for patient in patients_data[:1]:

    path = main_dir

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    print("Image Information. Total number of images:",len(slices))

    print(slices[0])





        

  

            

 




