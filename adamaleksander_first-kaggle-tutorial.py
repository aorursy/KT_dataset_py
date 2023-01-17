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
data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)
print(patients)

len(patients)
print(labels_df)
empty = labels_df.apply(lambda col: pd.isnull(col))

print(empty)
for patient in patients[:4]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    print(slices[0].pixel_array.shape, len(slices))

    
len(patients)
import matplotlib.pyplot as plt

import numpy as np

import cv2

import math



IMG_PX_SIZE = 150

HM_SLICES = 0



def chunks(l, n):

    for i in range(0, len(l), n):

        yield l[i:i +n]

 



def mean(l):

    return sum(l)/len(l)

for patient in patients[:2]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

   

    new_slices = []



    fig = plt.figure()

    for num, each_slice in enumerate(slices[:12]):

        y = fig.add_subplot(3,4, num+1)

        new_image = cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SIZE, IMG_PX_SIZE))

        y.imshow(new_image)

    plt.show()