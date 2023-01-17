# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import dicom
data_dir = '../input/sample_images/'
import os
patients = os.listdir(data_dir)
import pandas as pd
labels_df = pd.read_csv('../input/stage1_labels.csv', index_col=0)
labels_df.head()
for patient in patients[:3]:

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    print(slices[0].pixel_array.shape, len(slices))

    
label
len(patients)
import matplotlib.pyplot as plt

import cv2

import numpy as np

import math
IMG_PX_SIZE = 150

HM_SLICES = 20



for patient in patients[:10]:

    try:

        label = labels_df.get_value(patient, 'cancer')

        path = data_dir + patient

        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))



        new_slices = []

        slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

        chunk_sizes = math.ceil(len(slices) / HM_SLICES)

        for slice_chunk in chunks(slices, chunk_sizes):

            slice_chunk = list(map(mean, zip(*slice_chunk)))

            new_slices.append(slice_chunk)



        if len(new_slices) == HM_SLICES-1:

            new_slices.append(new_slices[-1])



        if len(new_slices) == HM_SLICES-2:

            new_slices.append(new_slices[-1])

            new_slices.append(new_slices[-1])



        if len(new_slices) == HM_SLICES+2:

            new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))

            del new_slices[HM_SLICES]

            new_slices[HM_SLICES-1] = new_val



        if len(new_slices) == HM_SLICES+1:

            new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))

            del new_slices[HM_SLICES]

            new_slices[HM_SLICES-1] = new_val



        print(len(slices), len(new_slices))



        fig = plt.figure()

        for ix, slice in enumerate(new_slices):

            y = fig.add_subplot(4,5, ix+1)

            plt.imshow(slice)

        plt.show()

    except:

        pass

    
def chunks(l, n):

    # Credit: Ned Batchelder

    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    """Yield successive n-sized chunks from l."""

    for i in range(0, len(l), n):

        yield l[i:i + n]



def mean(l):

    return sum(l) / len(l)
def chunks(l, nr_of_chunks=HM_SLICES):

    i = 0

    chunk_size = len(l) / nr_of_chunks

    while chunk_size * i < len(l):

        yield l[math.floor(chunk_size * i):math.floor(chunk_size * (i + 1))]

        i += 1
much_data = []

for num,patient in enumerate(patients):

    if num % 10 == 0:

        print(num)

    try:

        img_data,label = process_data(patient,labels_df,img_px_size=IMG_PX_SIZE, hm_slices=HM_SLICES)

        #print(img_data.shape,label)

        much_data.append([img_data,label])

    except KeyError as e:

        print('This is unlabeled data!')



np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE,IMG_PX_SIZE,HM_SLICES), much_data)
def process_data(patient,labels_df,img_px_size=50, hm_slices=20, visualize=False):

    

    label = labels_df.get_value(patient, 'cancer')

    path = data_dir + patient

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))



    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]

    

    chunk_sizes = math.ceil(len(slices) / hm_slices)

    for slice_chunk in chunks(slices, chunk_sizes):

        slice_chunk = list(map(mean, zip(*slice_chunk)))

        new_slices.append(slice_chunk)



    if len(new_slices) == hm_slices-1:

        new_slices.append(new_slices[-1])



    if len(new_slices) == hm_slices-2:

        new_slices.append(new_slices[-1])

        new_slices.append(new_slices[-1])



    if len(new_slices) == hm_slices+2:

        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))

        del new_slices[hm_slices]

        new_slices[hm_slices-1] = new_val

        

    if len(new_slices) == hm_slices+1:

        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))

        del new_slices[hm_slices]

        new_slices[hm_slices-1] = new_val



    if visualize:

        fig = plt.figure()

        for num,each_slice in enumerate(new_slices):

            y = fig.add_subplot(4,5,num+1)

            y.imshow(each_slice, cmap='gray')

        plt.show()



    if label == 1: label=np.array([0,1])

    elif label == 0: label=np.array([1,0])

        

    return np.array(new_slices),label
ls
# to be continued : after learning 3d-convnet. 

# thank  you sentdex!!