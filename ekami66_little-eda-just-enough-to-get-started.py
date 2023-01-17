import pandas as pd

import os



from PIL import Image

import cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



%matplotlib inline
def download_dataset():

    """

    Downloads the dataset and return the input paths

    :return: [train_data, test_data, metadata_csv, train_masks_csv, train_masks_data]

    """

    competition_name = "carvana-image-masking-challenge"



    destination_path = "../input/"

    files = ["train.zip", "test.zip", "metadata.csv.zip", "train_masks.csv.zip", "train_masks.zip"]

    datasets_path = ["../input/train", "../input/test", "../input/metadata.csv", "../input/train_masks.csv",

                    "../input/train_masks"]

    is_datasets_present = True



    # If the folders already exists then the files may already be extracted

    # This is a bit hacky but it's sufficient for our needs

    for dir_path in datasets_path:

        if not os.path.exists(dir_path):

            is_datasets_present = False



    if not is_datasets_present:

        pass

        #

        # I usually download my dataset with my home made tool on my local PC:

        # https://github.com/EKami/kaggle-data-downloader

        # But here on Kaggle we already have all the dataset present.

        #

        # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively

        # downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)

        #

        # for file in files:

        #    output_path = downloader.download_dataset(file, destination_path)

        #    downloader.decompress(output_path, destination_path)

        #    os.remove(output_path)

    else:

        print("All datasets are present.")

        

    return datasets_path
train_data, test_data, metadata_csv, train_masks_csv, train_masks_data = download_dataset()
metadata_df = pd.read_csv(metadata_csv)

train_masks_df = pd.read_csv(train_masks_csv)
metadata_df.head()
train_masks_df.head()
train_files = os.listdir(train_data)

test_files = os.listdir(test_data)

train_masks_files = os.listdir(train_masks_data)

print("Train files count: {}\nTrain masks files count: {}\nTest files count: {}"

      .format(len(train_files), len(train_masks_files), len(test_files)))
train_ids = list(set(t.split("_")[0] for t in train_files))

masks_ids = list(set(t.split("_")[0] for t in train_masks_files))

test_ids = list(set(t.split("_")[0] for t in test_files))



print("Train files unique ids count: {}\nTest files unique ids count: {}".format(len(train_ids), len(test_ids)))

assert len(train_ids) * 16 == len(train_files)

assert len(test_ids) * 16 == len(test_files)
def get_car_image_files(car_image_id, get_mask=False):

    if get_mask:

        if car_image_id in masks_ids:

            return [train_masks_data + "/" + s for s in train_masks_files if car_image_id in s]

        else:

            raise Exception("No mask with this ID found")

    elif car_image_id in train_ids:

        return [train_data + "/" + s for s in train_files if car_image_id in s]

    elif car_image_id in test_ids:

        return [test_data + "/" + s for s in test_files if car_image_id in s]

    raise Exception("No image with this ID found")

    

def get_image_matrix(image_path):

    img = Image.open(image_path)

    return np.asarray(img, dtype=np.uint8)
image_id = train_ids[0]



plt.figure(figsize=(20, 20))

img = get_image_matrix(get_car_image_files(image_id)[0])

mask = get_image_matrix(get_car_image_files(image_id, True)[0])

img_masked = cv2.bitwise_and(img, img, mask=mask)



print("Image shape: {} | image type: {} | mask shape: {} | mask type: {}"

      .format(img.shape, img.dtype, mask.shape, mask.dtype) )



plt.subplot(131)

plt.imshow(img)

plt.subplot(132)

plt.imshow(mask)

plt.subplot(133)

plt.imshow(img_masked);
def rle_encode(mask_image):

    pixels = mask_image.flatten()

    # We avoid issues with '1' at the start or end (at the corners of 

    # the original image) by setting those pixels to '0' explicitly.

    # We do not expect these to be non-zero for an accurate mask, 

    # so this should not harm the score.

    pixels[0] = 0

    pixels[-1] = 0

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2

    runs[1::2] = runs[1::2] - runs[:-1:2]

    return runs



def rle_to_string(runs):

    return ' '.join(str(x) for x in runs)



file_name = get_car_image_files(image_id)[0].split("/")[-1]

mask_rle = train_masks_df[train_masks_df['img'] == file_name]["rle_mask"].iloc[0]

assert rle_to_string(rle_encode(mask)) == mask_rle, "Mask rle don't match"

print("Mask rle match!")
images_path = [get_car_image_files(id) for id in train_ids[:5]]



for i, angles in enumerate(images_path):

    _, axs = plt.subplots(4, 4, figsize=(14, 10))  #  figsize=(20, 20)

    plt.rc('axes', grid=False)

    plt.subplots_adjust(wspace=0, hspace=0)

    axs = axs.ravel()

    

    for j, img_path in enumerate(angles):

        img = mpimg.imread(img_path)

        axs[j].axis('off')

        axs[j].imshow(img);
plt.figure(figsize=(12, 10))

sns.countplot(y="make", data=metadata_df);