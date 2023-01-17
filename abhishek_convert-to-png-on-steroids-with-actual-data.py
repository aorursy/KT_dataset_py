import os

import cv2

import glob2

import pydicom

from joblib import Parallel, delayed

from tqdm import tqdm_notebook as tqdm
def convert_images(filename, outdir):

    ds = pydicom.read_file(str(filename))

    img = ds.pixel_array

    img = cv2.resize(img, (128, 128))

    cv2.imwrite(outdir + filename.split('/')[-1][:-4] + '.png', img)
train_path = '../input/siim-dicom-images/siim-original/dicom-images-train/'

test_path = '../input/siim-dicom-images/siim-original/dicom-images-test/'
train_out_path = 'train_png/'

test_out_path = 'test_png/'
if not os.path.exists(train_out_path):

    os.makedirs(train_out_path)
if not os.path.exists(test_out_path):

    os.makedirs(test_out_path)
train_dcm_list = glob2.glob(os.path.join(train_path, '**/*.dcm'))

test_dcm_list = glob2.glob(os.path.join(test_path, '**/*.dcm'))
res1 = Parallel(n_jobs=8, backend='threading')(delayed(

    convert_images)(i, train_out_path) for i in tqdm(train_dcm_list[:100], total=len(train_dcm_list)))
res2 = Parallel(n_jobs=8, backend='threading')(delayed(

    convert_images)(i, test_out_path) for i in tqdm(test_dcm_list[:100], total=len(test_dcm_list)))