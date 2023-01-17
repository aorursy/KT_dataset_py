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
import dicom # for reading dicom files

import os # for doing directory operations

import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)



# Change this to wherever you are storing your data:

# IF YOU ARE FOLLOWING ON KAGGLE, YOU CAN ONLY PLAY WITH THE SAMPLE DATA, WHICH IS MUCH SMALLER



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)



labels_df.head()



for patient in patients[:1]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient



    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    print(len(slices), label)

    print(slices[0])
print(len(patients))

for patient in patients[:3]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    

    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    print(slices[0].pixel_array.shape, len(slices))
import matplotlib.pyplot as plt

import cv2

import numpy as np



for patient in patients[:1]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    

    fig = plt.figure()

    for num,each_slice in enumerate(slices[:12]):

        y = fig.add_subplot(3,4,num+1)

        new_img = cv2.resize(np.array(each_slice.pixel_array), (150,150))

        y.imshow(new_img)

    plt.show()

    
import matplotlib.pyplot as plt

import cv2

import numpy as np

import math

import dicom # for reading dicom files

import os # for doing directory operations

import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)





def chunks(l,n):

    for i in range(0, len(l), n):

        yield l[i:i+n]



def mean(l):

    return sum(l)/len(l)



IMG_PX_SLICE = 150

HM_SLICES = 20



data_dir = '../input/sample_images/'

patients = os.listdir(data_dir)

labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)



for patient in patients[:10]:

    try:

        label = labels_df.get_value(patient, 'cancer')

        path = data_dir + patient

        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

        slices = [cv2.resize(np.array(each_slice.pixel_array), (IMG_PX_SLICE,IMG_PX_SLICE)) for each_slice in slices]

        chunk_sizes = math.ceil(len(slices)/HM_SLICES)

        print('%d-%d'%(len(slices), chunk_sizes))

        new_slices = []

        

    except:

        pass

    

   