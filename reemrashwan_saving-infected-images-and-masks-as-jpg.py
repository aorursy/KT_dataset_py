# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames[:5]:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

import tqdm

import sys

import pydicom

import cv2

from PIL import Image

import matplotlib.pyplot as plt
sys.path.append("/kaggle/input/siim-acr-pneumothorax-segmentation/")

from mask_functions import mask2rle, rle2mask
train_images_path = "/kaggle/input/siim-train-test/dicom-images-train"

test_images_path = "/kaggle/input/siim-train-test/dicom-images-test/_/_/"
masks_data_path = "/kaggle/input/siim-train-test/train-rle.csv"



masks_data = pd.read_csv(masks_data_path)

masks_data.info()
masks_data.rename(columns={' EncodedPixels': 'EncodedPixels'}, inplace=True)
masks_data.head()
print(f"Length of masks data-frame: {len(masks_data)}")

print(f"Length of images in training dir: {len(os.listdir(train_images_path))}")

print(f"Length of images in testing dir: {len(os.listdir(test_images_path))}")
print(f"Unique images in masks dataframe: {masks_data.ImageId.nunique()}")
masks_data.info()
dataset = masks_data[masks_data.EncodedPixels != '-1'].reset_index(drop=True)

dataset.head()
dataset.info()
print(dataset.ImageId[0])

# print(dataset.ImagePath[0].split('/')[-1])
sample_mask = rle2mask(dataset.iloc[dataset.index[0]]['EncodedPixels'], 1024, 1024)

sample_image = pydicom.dcmread(dataset.iloc[dataset.index[0]]['ImagePath']).pixel_array

plt.imshow(sample_mask, cmap='gray')

plt.imshow(sample_mask.T)
plt.imshow(sample_image, cmap=plt.cm.bone)
plt.imshow(sample_image + sample_mask.T * 0.4, cmap='gray')
sample_mask.T.shape
sample_mask.T.dtype
def show_images(images, cols = 1, titles = None):

    """Display a list of images in a single figure with matplotlib.

    

    Parameters

    ---------

    images: List of np.arrays compatible with plt.imshow.

    

    cols (Default = 1): Number of columns in figure (number of rows is 

                        set to np.ceil(n_images/float(cols))).

    

    titles: List of titles corresponding to each image. Must have

            the same length as titles.

    """

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.axis('off')

        plt.imshow(image)

        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.show()
def plot_sample_images_and_masks(samples):

    images = []

    titles = []

    for i in range(len(samples)):

        image = pydicom.dcmread(samples.iloc[i].ImagePath).pixel_array

        mask = rle2mask(samples.iloc[i].EncodedPixels, 1024, 1024).T

        masked_image = image + mask * 0.4

        

        images.extend([image, mask, masked_image])

        titles.extend([f"{i} image", f"{i} mask", f"{i} masked_image"])

        

    show_images(images=images, titles=titles, cols=3)

    
samples = dataset[dataset.EncodedPixels != '-1'].sample(3).reset_index(drop=True)

samples
plot_sample_images_and_masks(samples)
IMG_WIDTH = 254

IMG_HEIGHT = 254
# test on a sample

# m1 = dataset.Mask.values[0]

# m2 = dataset.Mask.values[22]

# m3 = np.sum(m1, m2)

# plt.imshow(m3)

# plt.imshow(np.clip(m3, 0, 255))
# extracting masks

# dataset['Mask'] = dataset.EncodedPixels.apply(lambda cell_value: cv2.resize(rle2mask(cell_value, 1024, 1024).T, (IMG_WIDTH, IMG_HEIGHT)))

dataset['Mask'] = dataset.EncodedPixels.apply(lambda cell_value: rle2mask(cell_value, 1024, 1024).T)

dataset.head()
dataset.head()
unique_ids_count = dataset[['ImageId', 'Mask']].groupby(['ImageId']).Mask.count()  # don't use small c in count

ids_with_one_mask = unique_ids_count[unique_ids_count == 1].reset_index()

ids_with_more_than_one_mask = unique_ids_count[unique_ids_count > 1].reset_index()

# ids_with_more_than_one_mask

# ids_with_one_mask
one_mask_data = pd.merge(dataset[['ImageId', 'Mask']], ids_with_one_mask.ImageId, on='ImageId', how='right')

more_than_one_mask_data = pd.merge(dataset[['ImageId', 'Mask']], ids_with_more_than_one_mask.ImageId, on='ImageId', how='right')
more_than_one_mask_data
combined_masks =  more_than_one_mask_data[['ImageId', 'Mask']].groupby('ImageId').Mask.apply(np.sum)

combined_masks = combined_masks.apply(np.clip, a_min=0, a_max=1).reset_index()  # maybe some masks are overlapped, so when sum it will exceed 255
plt.imshow(combined_masks.Mask.sample(1).values[0])
# combined_masks.head()

# one_mask_data.head()



final_dataset = pd.concat([combined_masks, one_mask_data], ignore_index=True)

final_dataset
images_paths = glob.glob(f"{train_images_path}/*/*/*.dcm")

print(len(images_paths))

# print(images_paths[:5])
def get_image_id(image_path):

    image_name = image_path.rsplit("/", maxsplit=1)[-1]

    return image_name[:-4]  # id without extension



def get_images_path_and_id(images_paths):

    images_data = pd.DataFrame(columns=['ImageId', 'ImagePath'])

#     for image_path in tqdm.notebook.tqdm(images_paths):

    for image_path in tqdm.tqdm_notebook(images_paths):

        image_id = get_image_id(image_path)

        images_data = images_data.append({'ImageId': image_id, 'ImagePath': image_path}, ignore_index=True)

        

    return images_data



def get_images_path_and_id(images_paths):

    images_data = pd.DataFrame(columns=['ImageId', 'ImagePath'])

    images_data = {"ImageId": [], "ImagePath": []}

#     for image_path in tqdm.notebook.tqdm(images_paths):

    for image_path in tqdm.tqdm_notebook(images_paths):

        image_id = get_image_id(image_path)

        images_data['ImageId'].append(image_id)

        images_data['ImagePath'].append(image_path)

        

    images_data = pd.DataFrame(images_data)

    

    return images_data

images_paths_and_ids = get_images_path_and_id(images_paths)
images_paths_and_ids.head()
final_dataset.info()
final_dataset = pd.merge(images_paths_and_ids, final_dataset, on='ImageId', how='inner')

final_dataset.info()
final_dataset.head()
# arrange dataset 

final_dataset = final_dataset.sort_values('ImageId', ascending=True).reset_index(drop=True)

final_dataset.head()
def create_dir(dirname):

    try:

        os.makedirs(dirname)

        print(f"Directory '{dirname}' created.") 

    except FileExistsError:

        print(f"Directory '{dirname}' already exists.") 





# create a train directory

train_images_dir = '/kaggle/working/train/images'

train_masks_dir = '/kaggle/working/train/masks'



create_dir(train_images_dir)

create_dir(train_masks_dir)
def extract_dcm_to_folder_as_jpg(dcm_id, dcm_path, destination_path):

    # read image

    dcm_file = pydicom.dcmread(dcm_path)

    # to np array

    image_array = dcm_file.pixel_array

    save_np_array_as_image(dcm_id, image_array, destination_path)



    

def save_np_array_as_image(image_id, image_array, destination_path):

    image = convert_image_array_to_pil_image(image_array).convert('L')

     # save to disk

    image.save(os.path.join(destination_path, f"{image_id}.jpg"))

    

    

def convert_image_array_to_pil_image(image_array):

    image = Image.fromarray(image_array)  # to gray

    return image
# save dcm as jpg

final_dataset.apply(lambda row: extract_dcm_to_folder_as_jpg(row['ImageId'], row['ImagePath'], train_images_dir), axis=1)

print(f"Done saving {len(os.listdir(train_images_dir))} images.")

# tqdm.tqdm_notebook.pandas()

# dataset.progress_apply(lambda row: extract_dcm_to_folder_as_jpg(row['ImageId'], row['ImagePath'], train_images_dir), axis=1)



final_dataset.apply(lambda row: save_np_array_as_image(row['ImageId'], row['Mask'], train_masks_dir), axis=1)

print(f"Done saving {len(os.listdir(train_masks_dir))} masks.")
sample = final_dataset.sample(1)

sample_path = os.path.join(train_masks_dir, sample.ImageId.values[0]) + '.jpg'

sample_mask_image = Image.open(sample_path)

sample_mask_array = sample.Mask.values[0]



print("Array")

plt.imshow(sample_mask_array)
print("Saved image")

plt.imshow(sample_mask_image)