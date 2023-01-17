import cv2

import gc

import io

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time

from tqdm import tqdm_notebook as tqdm

import zipfile

import warnings

warnings.filterwarnings("ignore")
# To support Multiprocessing

import multiprocessing
HEIGHT = 137

WIDTH = 236

SIZE = 128



TRAIN = ['/kaggle/input/bengaliai-cv19/train_image_data_0.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_1.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_2.parquet',

         '/kaggle/input/bengaliai-cv19/train_image_data_3.parquet']
def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size = SIZE, pad = 16):

    # Crop a box around pixels large than the threshold 

    # Some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    

    # Cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    

    # Remove low intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    

    # Make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode = 'constant')

    

    return cv2.resize(img, (size, size))
# Start timer

start_time = time.time()



with zipfile.ZipFile('train_single.zip', 'w') as img_out:

    for fname in TRAIN:

        # Read parquet file into pandas.

        df = pd.read_parquet(fname)

        

        # The input inverted

        data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        for idx in tqdm(range(len(df))):

            name = df.iloc[idx,0]

            

            # Normalize each image by its max val

            img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)

            img = crop_resize(img)

        

            img = cv2.imencode('.png',img)[1]

            img_out.writestr(name + '.png', img)

            

# Total time

print('Processing time standard: {0} [sec]'.format(time.time() - start_time))
def process_image_multi_v1(data, name):

    # Reshape data

    data = data.reshape(HEIGHT, WIDTH)

    

    # Normalize each image by its max val

    data = 255 - data.astype(np.uint8)

    img = (data*(255.0/data.max())).astype(np.uint8)

    

    # Crop, Resize and encode as PNG file

    img = crop_resize(img)

    img = cv2.imencode('.png',img)[1]

    

    return name, img
# CPU core count

cpu_count = multiprocessing.cpu_count()

print(cpu_count)
# Start Multi Processing

start_multi_time_v1 = time.time()



try:

    # Setup multiprocessing pool

    pool = multiprocessing.Pool(processes = cpu_count)



    with zipfile.ZipFile('train_multi.zip', 'w') as img_out:

        for fname in TRAIN:

            # Read Parquet file

            df = pd.read_parquet(fname)

            

            # Prep the input for pool.starmap.

            data = df.iloc[:, 1:].values

            names = df.image_id.tolist()



            for name, img in pool.starmap(process_image_multi_v1, zip(data, names)):

                img_out.writestr(name + '.png', img)



finally:

    pool.close()

    pool.join()



# Total time

print('Processing time multi v1: {0} [sec]'.format(time.time() - start_multi_time_v1))
def process_image_multi_v2(data, name):

    # Reshape data

    data = data.reshape(HEIGHT, WIDTH)

    

    # Normalize each image by its max val

    data = 255 - data.astype(np.uint8)

    img = (data*(255.0/data.max())).astype(np.uint8)

    

    # Crop, Resize and encode as PNG file

    img = crop_resize(img)

    img = cv2.imencode('.png',img)[1]

    

    # Save image

    cv2.imwrite('./subdir/' + name + '.png', img)

    

# Start Multi Processing

start_multi_time_v2 = time.time()



try:

    pool = multiprocessing.Pool(processes = cpu_count)



    # Process Images Standard.

    for fname in TRAIN:

        # Read Parquet file

        df = pd.read_parquet(fname)



        # Invert the input

        data = df.iloc[:, 1:].values

        names = df.image_id.tolist()

        

        # Multi Process Images

        pool.starmap(process_image_multi_v2, zip(data, names))



finally:

    pool.close()

    pool.join()



# Total time

print('Processing time multi v2: {0} [sec]'.format(time.time() - start_multi_time_v2))