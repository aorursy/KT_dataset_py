import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom as dicom

import glob

import os

from tqdm import tqdm

from PIL import Image



traindir = "/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images"

testdir = "/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_test_images"



train_dicom_files = glob.glob(f"{traindir}/*.dcm")

print(f"Number of train files: {len(train_dicom_files)}")



test_dicom_files = glob.glob(f"{testdir}/*.dcm")

print(f"Number of test files: {len(test_dicom_files)}")
# Helper function adapted from: 

# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/



def get_pixels_hu(scan):

    image = np.stack(scan.pixel_array)

    

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 1

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    intercept = scan.RescaleIntercept

    slope = scan.RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)



def hu2pil(img):

    # as we want to detect hemorrages (HU between 0 to 100), we can normalizeas 0

    # -1000: air

    # -500: lung

    # (-100, -50): Fat

    # 0: Water

    # (+30, +70): Blood

    # (+10, +40): Muscle

    # (+40, +60): Liver

    # (+600, +3000): Bone

    

    # remove values greater than 3000 (out of scale)

    img = np.less_equal(img, 3000)*img

    # remove values greater lower than -1000 (out of scale)

    img = np.greater_equal(img, -1000)*img

    

    # convert to positive values (0 to 4000 range)

    img = img + 1000

    

    img = img/4000 # (0 to 1 scaling)

    

    img = Image.fromarray(np.uint8(img*255.0))

    

    return img
outputdir = "./output"

os.system(f"mkdir {outputdir} {outputdir}/train {outputdir}/test")



for file in tqdm(train_dicom_files):

    id = os.path.splitext(os.path.basename(file))[0]

    patient = dicom.read_file(file)

    img = get_pixels_hu(patient)

    img = hu2pil(img)

    img.save(f"{outputdir}/train/{id}.jpg")

    break # REMOVE WHEN USING IT (IN THIS SCRIPT THE LOOP IS DISABLED)

    

for file in tqdm(test_dicom_files):

    id = os.path.splitext(os.path.basename(file))[0]

    patient = dicom.read_file(file)

    img = get_pixels_hu(patient)

    img = hu2pil(img)

    img.save(f"{outputdir}/test/{id}.jpg")

    break # REMOVE WHEN USING IT (IN THIS SCRIPT THE LOOP IS DISABLED)
import hashlib, os



unique = dict()

duplicate = []

for filename in glob.glob(f"{outputdir}/*/*.jpg"):

    if os.path.isfile(filename):

        filehash = hashlib.md5(open(filename, 'rb').read()).hexdigest()



        if filehash not in unique: 

            unique[filehash] = filename

        else:

            print (f"{filename} is a duplicate of {unique[filehash]}")

            duplicate.append(filename)

print(f"Number of duplicates: {len(duplicate)}")

for file in duplicate:

    print(f"{file}")

    os.system(f"rm {file}")