import pydicom

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

import glob

import re

import os

import numpy as np

import pandas as pd



import pydicom

import matplotlib.pyplot as plt

from tqdm.auto import tqdm



import scipy

from scipy import ndimage



tqdm.pandas()
os.listdir('/kaggle/input/')
DATA_DIR = "/kaggle/input/rsna-str-pulmonary-embolism-detection"

#DATA_DIR = "data"

# !ls -l data/train | wc -l/
df_train = pd.read_csv(f"{DATA_DIR}/train.csv")

df_train.head()

df_test = pd.read_csv(f"{DATA_DIR}/test.csv")

df_test.head()

# !ls /kaggle/input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf



TRAIN_DIR = "/kaggle/input/rsna-str-pulmonary-embolism-detection/train"

flist1 = os.listdir(TRAIN_DIR)

dicom_list = []



for i in range(len(flist1)):

    temp_files = len(glob.glob(f'{TRAIN_DIR}/{flist1[i]}/*/*.dcm'))

    dicom_list.append(temp_files)
flist1[i]
dicom_list[0:10]
print(max(dicom_list), min(dicom_list))
import matplotlib.pyplot as plt

plt.plot(dicom_list)
files = glob.glob(f"{DATA_DIR}/train/0003b3d648eb/*/*.dcm")

# files
files.sort(key=lambda f: int(re.sub('\D', '', f)))

# files
import types



ds = pydicom.dcmread(f'{DATA_DIR}/train/0003b3d648eb/d2b2960c2bbf/2cb4facfafee.dcm')

attrs = dir(ds)

for attr in attrs:

    if attr.startswith("_"):

        continue

    if attr == "PixelData" or attr == "pixel_array":

        continue

    var_type = type(getattr(ds,attr))

    if var_type == types.MethodType:

        continue

    print(f"{attr}: {getattr(ds,attr)}")
import math

from collections.abc import Iterable

import pydicom



def show_img(img_path, colormap = None, extra_brightness=0):

    ds = pydicom.dcmread(img_path)

    shape = ds.pixel_array.shape

    target = 255



    # Convert to float to avoid overflow or underflow losses.

    image_2d = ds.pixel_array.astype(float)

    img_data = image_2d

    print(f"data min: {img_data.min()}, max: {img_data.max()}")

    print(f"window center: {ds.WindowCenter}, rescale intercept: {ds.RescaleIntercept}")

    multival = isinstance(ds.WindowCenter, Iterable)

    if multival:

        scale_center = -ds.WindowCenter[0]

    else:

        scale_center = -ds.WindowCenter

    intercept = scale_center+ds.RescaleIntercept+extra_brightness

    print(f"final intercept: {intercept}")

    image_2d += intercept

    print(f"after applying intercept, min: {image_2d.min()}, max: {image_2d.max()}")



    # Rescaling grey scale between 0-255

    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    print(f"after scaling to 0-255, min: {image_2d_scaled.min()}, max: {image_2d_scaled.max()}")



    # Convert to uint

    image_2d_scaled = np.uint8(image_2d_scaled)



    plt.figure(figsize=(12,8))

    plt.imshow(image_2d_scaled, cmap=colormap)

    plt.show()



#show_img(f'{DATA_DIR}/train/ID00011637202177653955184/1.dcm', colormap=plt.cm.bone) <-image 0 below

show_img(f'{DATA_DIR}/train/0003b3d648eb/d2b2960c2bbf/2cb4facfafee.dcm', colormap=plt.cm.bone) #image 1

show_img(files[0], colormap=plt.cm.bone) #image 2

import types



ds = pydicom.dcmread(files[0])

attrs = dir(ds)

for attr in attrs:

    if attr.startswith("_"):

        continue

    if attr == "PixelData" or attr == "pixel_array":

        #skip printing the long arrays as they will just spam the output too much with hex code

        continue

    var_type = type(getattr(ds,attr))

    if var_type == types.MethodType:

        continue

    print(f"{attr}: {getattr(ds,attr)}")
#show_img(f'{DATA_DIR}/train/ID00011637202177653955184/1.dcm', colormap=None)

show_img(f'{DATA_DIR}/train/0003b3d648eb/d2b2960c2bbf/2cb4facfafee.dcm', colormap=None)

show_img(files[0], colormap=None)
#show_img(f'{DATA_DIR}/train/ID00011637202177653955184/1.dcm', colormap=plt.cm.bone, extra_brightness=1000)

show_img(f'{DATA_DIR}/train/0003b3d648eb/d2b2960c2bbf/2cb4facfafee.dcm', colormap=plt.cm.bone, extra_brightness=1000)

show_img(files[0], colormap=plt.cm.bone, extra_brightness=1000)

def resample_z(image, scan, z, new_spacing=[1, 1, 1]):

    #This is probably a bit more complex implementation than is needed, since we just want to scale Z to constant

    #But its what I ended up and it works, so I left it as is..

    # Determine current pixel spacing

    spacing = np.array([scan[0]["SliceThickness"]] + [scan[0]["PixelSpacing_0"], scan[0]["PixelSpacing_1"]], dtype=np.float32)



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    #factor is (z,y,x)

    real_resize_factor = (z/image.shape[0],1,1)



    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')



    return image, new_spacing
def dump_scaled(np_arrays, df_name, patient_id, save_png=True, save3d=False):

    full_3d = []

    full_3d_256 = []

    full_3d_192 = []

    full_3d_128 = []

    for idx, img in enumerate(np_arrays):

        file = f"img{idx}"

        processed_filename = os.path.basename(file)

        processed_filename = processed_filename.split(".")[0]+".png"

        processed_dir = f"scaled_png/{df_name}/{patient_id}"

        os.makedirs(processed_dir, exist_ok=True)

        processed_path = f"{processed_dir}/{processed_filename}"



        im = Image.fromarray(img)

#        im = detect_border(im, file)

        if save_png:

            im.save(processed_path)

        full_3d.append(np.array(im))

#         im256 = im.resize((256,256))

#         im192 = im.resize((192,192))

#         im128 = im.resize((128,128))

#         full_3d_256.append(np.array(im256))

#         full_3d_192.append(np.array(im192))

#         full_3d_128.append(np.array(im128))

    full_3d = np.array(full_3d)

#     full_3d_256 = np.array(full_3d_256)

#     full_3d_192 = np.array(full_3d_192)

#     full_3d_128 = np.array(full_3d_128)

    if save3d:

        np.save(f"scaled_png/{df_name}/{patient_id}/full_3d_512", full_3d)

        #np.save(f"scaled_png/{df_name}/{patient_id}/full_3d_256", full_3d_256)

        #np.save(f"scaled_png/{df_name}/{patient_id}/full_3d_192", full_3d_192)

        #np.save(f"scaled_png/{df_name}/{patient_id}/full_3d_128", full_3d_128)

import os

from pydicom.pixel_data_handlers.util import apply_voi_lut

import cv2



#this function is similar to the one used higher above to illustrate the intercept scaling on color values

#this is just tailored to be run on all files at once

def scale_to_png(ds, file, df_name, patient_id):

    image_2d = ds.pixel_array.astype(float)

    multival = isinstance(ds.WindowCenter, Iterable)

    if multival:

        scale_center = -ds.WindowCenter[0]

    else:

        scale_center = -ds.WindowCenter

    intercept = scale_center+ds.RescaleIntercept

    image_2d += intercept



    # Rescaling grey scale between 0-255

    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    img_data = image_2d_scaled



    image_2d_scaled = np.uint8(image_2d_scaled)



    processed_filename = os.path.basename(file)

    processed_filename = processed_filename.split(".")[0]+".png"

    processed_dir = f"png/{df_name}/{patient_id}"

    os.makedirs(processed_dir, exist_ok=True)

    processed_path = f"{processed_dir}/{processed_filename}"



    im = Image.fromarray(image_2d_scaled)

    im = detect_border(im, file)

    im.save(processed_path)



    shape = ds.pixel_array.shape



    return im, processed_filename



from PIL import Image, ImageChops, ImageOps



#most common filesize seemed to be 512x512, so converting all files to that size

def resize_to_512(im: Image, image_name: str):

    width, height = im.size

    if width == 512 and height == 512:

        return im

    if width != height:

        if width < height:

            pad_w = height - width

            pad_w /= 2

            pad_h = 0

        else:

            pad_w = 0

            pad_h = width-height

            pad_h /= 2

        padding = (pad_w, pad_h, pad_w, pad_h )

        print("WARN: resizing image {image_name}")

        #we should not come here but if we do, this should resize to square

        ImageOps.expand(im, padding, Image.ANTIALIAS)



    im2 = im.resize((512, 512))

    return im2



#some images in the dataset have a grayish border around the actual image data, 

#with the actual image data being 512x512 size. 

#this removes the grey border and keeps the actual data

def detect_border(im, filepath):

    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))

    diff = ImageChops.difference(im, bg)

    diff = ImageChops.add(diff, diff, 1.0, -0)

    bbox = diff.getbbox()

    if bbox:

        im2 = im.crop(bbox)

        width, height = im2.size

        if width == height:

            im = im2

        else:

            im = im

    im = resize_to_512(im, filepath)

    return im



def process_dataset(base_dir, df_data, df_name):

    ds_files = []

    attribute_names = set()



    #patients = df_data["Patient"].unique()

    patients = df_data["StudyInstanceUID"].unique()

    processed_patients = set()

    num_patients = patients.shape[0]

    for idx, patient_id in tqdm(enumerate(patients), total=num_patients):

        print(f"processing patient: {patient_id}")

        if patient_id in processed_patients:

            continue

        processed_patients.add(patient_id)

        #the following is just to skip the first 5 images on Kaggle, as it does not have GDCM installed

        if len(processed_patients) < 5:

            continue

        patient_dir = f'{base_dir}/{patient_id}/'

        files = glob.glob(f"{patient_dir}/*/*.dcm")

        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        ds_size_before = len(ds_files)

        patient_images = []

        patient_meta = []

        for file in files:

            ds = pydicom.dcmread(file)

            attrs = dir(ds)

            metadata = {}

            metadata["PatientId"] = patient_id

            metadata["file_idx"] = idx

            ds_files.append(metadata)

            im, filename = scale_to_png(ds, file, df_name, patient_id)

            metadata["filename"] = filename

            patient_images.append(np.array(im))

            patient_meta.append(metadata)



            for attr in attrs:

                if attr.startswith("_"):

                    continue

                if attr == "PixelData" or attr == "pixel_array" or attr == "fromkeys" or attr == "copy":

                    continue

                var_type = type(getattr(ds,attr))

                if var_type == types.MethodType:

                    continue

                if attr not in attribute_names:

                    print(f"{attr}: {var_type}")

                    attribute_names.add(attr)

                value = getattr(ds,attr)

                if type(value) is list or type(value) is pydicom.multival.MultiValue:

                    for sub_idx, sub_value in enumerate(value):

                        metadata[f"{attr}_{sub_idx}"] = f"{sub_value}"

                else:

                    metadata[attr] = f"{value}"

            del ds

            del attrs

        np_images = np.array(patient_images)

        resampled_pixels_64, spacing = resample_z(np_images, patient_meta, 64)

        resampled_pixels_200, spacing = resample_z(np_images, patient_meta, 200)

        #the PNG files are not really needed outside checking the code works, since the 3D arrays contain it all

        #but I leave it here so we can see the results

        dump_scaled(resampled_pixels_64, df_name+"_64", patient_id, save_png=True, save3d=True)

        dump_scaled(resampled_pixels_200, df_name+"_200", patient_id, save_png=True, save3d=True)

        ds_size_after = len(ds_files)

        ds_diff = ds_size_after - ds_size_before

        #this is here just to cap the number of files processed in Kaggle

        if len(processed_patients) > 10:

            break

    return ds_files



base_dir = f'{DATA_DIR}/train/'

ds_train_files = process_dataset(base_dir, df_train, "train")



base_dir = f'{DATA_DIR}/test/'

ds_test_files = process_dataset(base_dir, df_test, "test")



#NOTE: below all different attributes and their types found in the dataset will be printed as well
# !ls png/train
#also check the 3D arrays. here we generated ones with 20 and 30 images, so should see separate dirs for each

# !ls scaled_png/
# !ls scaled_png/train_30/
# df_test_file_meta = pd.DataFrame(ds_test_files)

# df_test_file_meta.to_csv("meta_files_test.csv")

# df_test_file_meta.head()

# df_train_file_meta = pd.DataFrame(ds_train_files)

# df_train_file_meta.to_csv("meta_files_train.csv")

# df_train_file_meta.head()