# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pydicom

import matplotlib.pyplot as plt

import pandas as pd

import sys

import glob

from scipy.ndimage import zoom

from tqdm import tqdm

import gc



dir_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'
from pydicom.pixel_data_handlers.util import apply_modality_lut

# from pydicom.pixel_data_handlers.util import apply_color_lut

for fname in glob.glob('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00355637202295106567614' + '/*dcm', recursive=False):

    print(fname)

    ttt = pydicom.dcmread(fname)

    print(ttt)

    print('ppp', ttt.pixel_array.max(), ttt.pixel_array.min())

    hu = apply_modality_lut(ttt.pixel_array, ttt)

    print(hu.max(), hu.min())

    break
def dicom2d_to_3d(path):

    # load the DICOM files

    files = []

    for fname in glob.glob(path + '/*dcm', recursive=False):

        files.append(pydicom.dcmread(fname))



    # skip files with no SliceLocation (eg scout views)

    slices = []

    skipcount = 0

    for f in files:

        slices.append(f)



    # ensure they are in the correct order

    # slices = sorted(slices, key=lambda s: s.SliceLocation)

    slices = sorted(slices, key=lambda s: s[0x00200013].value/len(files))





    # pixel aspects, assuming all slices are the same

    ps = slices[0].PixelSpacing

    ss = len(files)/len(files)#slices[0].SliceThickness

    ax_aspect = ps[1]/ps[0]

    sag_aspect = ps[1]/ss

    cor_aspect = ss/ps[0]



    # create 3D array

    img_shape = list(slices[0].pixel_array.shape)

    img_shape.append(len(slices))

    img3d = np.zeros(img_shape)



    # fill 3D array with the images from the files

    for i, s in enumerate(slices):

        img2d = s.pixel_array

        img3d[:, :, i] = img2d



    

    resize_img3d = zoom(img3d, (64/files[0].Rows, 64/files[0].Columns, 64/len(files)))

    # normalization

    #resize_img3d = resize_img3d.clip(0, resize_img3d.max()) / resize_img3d.max() * 255

    resize_img3d = ((resize_img3d - resize_img3d.min()) / (resize_img3d.max() - resize_img3d.min())) * 255

    resize_img3d = resize_img3d.astype('int8')

    

    del files

    del img3d



    return resize_img3d.astype('uint8')
train_data = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

test_data = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

sub = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
dir_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'

patient_name = 'ID00111637202210956877205'

train_image_dict = {}

except_patient_name = []

for patient_name in tqdm(train_data.Patient.unique()):

    try:

        train_image_dict[patient_name] = dicom2d_to_3d(dir_path + patient_name)

        np.save('/kaggle/working/' + patient_name + '.npy', train_image_dict[patient_name])

    except:

        except_patient_name.append(patient_name)    
test = np.load('ID00111637202210956877205.npy')
plt.imshow(test[:, :, 34])