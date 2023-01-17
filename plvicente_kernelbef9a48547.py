# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from glob import glob 

import matplotlib.pyplot as plt



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

def show_dcm_info(dataset):

    print("Filename.........:", file_path)

    print("Storage type.....:", dataset.SOPClassUID)

    print()



    pat_name = dataset.PatientName

    display_name = pat_name.family_name + ", " + pat_name.given_name

    print("Patient's name......:", display_name)

    print("Patient id..........:", dataset.PatientID)

    print("Patient's Age.......:", dataset.PatientAge)

    print("Patient's Sex.......:", dataset.PatientSex)

    print("Modality............:", dataset.Modality)

    print("Body Part Examined..:", dataset.BodyPartExamined)

    print("View Position.......:", dataset.ViewPosition)

    

    if 'PixelData' in dataset:

        rows = int(dataset.Rows)

        cols = int(dataset.Columns)

        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(

            rows=rows, cols=cols, size=len(dataset.PixelData)))

        if 'PixelSpacing' in dataset:

            print("Pixel spacing....:", dataset.PixelSpacing)



def plot_pixel_array(dataset, figsize=(10,10)):

    plt.figure(figsize=figsize)

    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

    plt.show()

    

#Show some images

from glob import glob

train_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))

test_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-test/*/*/*.dcm'))



print(len(train_fns))

print(len(test_fns))
for file_path in sorted(glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')):

    dataset = pydicom.dcmread(file_path)

    show_dcm_info(dataset)

    plot_pixel_array(dataset)

    break # Comment this out to see all
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
#num_img = len(glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm'))

fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')):

    dataset = pydicom.dcmread(file_path)

    #show_dcm_info(dataset)

    

    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)
df = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample images/train-rle-sample.csv', header=None, index_col=0)



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in sorted(glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)

    src_img = dataset.pixel_array

    img_clahe = clahe.apply(src_img)

    ax[q].imshow(img_clahe, cmap=plt.cm.bone)

    if df.loc[file_path.split('/')[-1][:-4],1] != '-1':

        mask = rle2mask(df.loc[file_path.split('/')[-1][:-4],1], 1024, 1024).T

        ax[q].set_title('See Marker')

        ax[q].imshow(mask, alpha=0.3, cmap="Reds")

    else:

        ax[q].set_title('Nothing to see')
start = 5   # Starting index of images

num_img = 4 # Total number of images to show



fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

for q, file_path in enumerate(glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):

    dataset = pydicom.dcmread(file_path)

    #show_dcm_info(dataset)

    

    ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)