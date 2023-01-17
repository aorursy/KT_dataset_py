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
import dicom

import os

import pandas as pd

from glob import glob



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

label_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)



for patient in patients[:2]:

    label = label_df.get_value(patient, 'cancer')

    #print('%s-cancer[%d]'%(patient, label))

    path = data_dir+patient

    

    slices = [dicom.read_file(path+'/'+s) for s in os.listdir(path)]

    print(len(slices),label)

    print(slices[0])

    

    