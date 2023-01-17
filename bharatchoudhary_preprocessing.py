import pandas as pd
from PIL import Image, ImageFile
from scipy import ndimage

import os
from time import time
from joblib import Parallel, delayed
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import pydicom
from scipy import ndimage
from PIL import Image, ImageFile
import matplotlib.pylab as plt
from tqdm import tqdm_notebook, tqdm
%matplotlib inline

base_url = '/home/ubuntu/kaggle/rsna-intracranial-hemorrhage-detection/'
TRAIN_DIR = '/home/ubuntu/kaggle/rsna-intracranial-hemorrhage-detection/stage_2_train'
TEST_DIR = '/home/ubuntu/kaggle/rsna-intracranial-hemorrhage-detection/stage_2_test'
os.listdir(base_url)

def prepare_dicom(dcm, width=None, level=None, norm=True):
    """
    Converts a DICOM object to a 16-bit Numpy array (in Hounsfield units)
    :param dcm: DICOM Object
    :return: Numpy array in int16
    """

    try:
        # https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        if dcm.BitsStored == 12 and dcm.PixelRepresentation == 0 and dcm.RescaleIntercept > -100:
            x = dcm.pixel_array + 1000
            px_mode = 4096
            x[x >= px_mode] = x[x >= px_mode] - px_mode
            dcm.PixelData = x.tobytes()
            dcm.RescaleIntercept = -1000

        pixels = dcm.pixel_array.astype(np.float32) * dcm.RescaleSlope + dcm.RescaleIntercept
    except ValueError as e:
        print("ValueError with", dcm.SOPInstanceUID, e)
        return np.zeros((512, 512))

    # Pad the image if it isn't square
    if pixels.shape[0] != pixels.shape[1]:
        (a, b) = pixels.shape
        if a > b:
            padding = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            padding = (((b - a) // 2, (b - a) // 2), (0, 0))
        pixels = np.pad(pixels, padding, mode='constant', constant_values=0)
        
    if not width:
        width = dcm.WindowWidth
        if type(width) != pydicom.valuerep.DSfloat:
            width = width[0]
    if not level:
        level = dcm.WindowCenter
        if type(level) != pydicom.valuerep.DSfloat:
            level = level[0]
    lower = level - (width / 2)
    upper = level + (width / 2)
    img = np.clip(pixels, lower, upper)

    if norm:
        return (img - lower) / (upper - lower)
    else:
        return img
class CropHead(object):
    def __init__(self, offset=10):
        """
        Crops the head by labelling the objects in an image and keeping the second largest object (the largest object
        is the background). This method removes most of the headrest

        Originally made as a image transform for use with PyTorch, but too slow to run on the fly :(
        :param offset: Pixel offset to apply to the crop so that it isn't too tight
        """
        self.offset = offset

    def crop_extents(self, img):
        try:
            if type(img) != np.array:
                img_array = np.array(img)
            else:
                img_array = img

            labeled_blobs, number_of_blobs = ndimage.label(img_array)
            blob_sizes = np.bincount(labeled_blobs.flatten())
            head_blob = labeled_blobs == np.argmax(blob_sizes[1:]) + 1  # The number of the head blob
            head_blob = np.max(head_blob, axis=-1)

            mask = head_blob == 0
            rows = np.flatnonzero((~mask).sum(axis=1))
            cols = np.flatnonzero((~mask).sum(axis=0))

            x_min = max([rows.min() - self.offset, 0])
            x_max = min([rows.max() + self.offset + 1, img_array.shape[0]])
            y_min = max([cols.min() - self.offset, 0])
            y_max = min([cols.max() + self.offset + 1, img_array.shape[1]])

            return x_min, x_max, y_min, y_max
        except ValueError:
            return 0, 0, -1, -1

    def __call__(self, img):
        """
        Crops a CT image to so that as much black area is removed as possible
        :param img: PIL image
        :return: Cropped image
        """

        x_min, x_max, y_min, y_max = self.crop_extents(img)

        try:
            if type(img) != np.array:
                img_array = np.array(img)
            else:
                img_array = img

            return Image.fromarray(np.uint8(img_array[x_min:x_max, y_min:y_max]))
        except ValueError:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(offset={})'.format(self.offset)
crop_head = CropHead()
def dcm_to_png(row, image_dirs, dataset, width, level, crop, crop_head, output_path):
    r_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["red"] + ".dcm"))
    g_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["green"] + ".dcm"))
    b_dcm = pydicom.dcmread(os.path.join(image_dirs[dataset], row["blue"] + ".dcm"))
    r = prepare_dicom(r_dcm, width, level)
    g = prepare_dicom(g_dcm, width, level)
    b = prepare_dicom(b_dcm, width, level)
    img = np.stack([r, g, b], -1)
    img = (img * 255).astype(np.uint8)
    im = Image.fromarray(img)

    if crop:
        im = crop_head(im)

    im.save(os.path.join(output_path, row["green"] + ".png"))
def prepare_png_images(dataset, folder_name, width=None, level=None, crop=True):
    start = time()

    triplet_dfs = {
        "train": os.path.join(base_url, "train_triplets.csv"),
        "test_stage_2": os.path.join(base_url, "stage_2_test_triplets.csv")
    }

    image_dirs = {
        "train": os.path.join(base_url, "stage_2_train"),
        "test_stage_2": os.path.join(base_url, "stage_2_test")
    }

    output_path = os.path.join(base_url, "png", dataset, f"{folder_name}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    triplets = pd.read_csv(triplet_dfs[dataset])
    crop_head = CropHead()
    for _, row in tqdm(triplets.iterrows()):
        dcm_to_png(row, image_dirs, dataset, width, level, crop, crop_head, output_path)


    print("Done in", (time() - start) // 60, "minutes")

prepare_png_images("train", "adjacent-brain-cropped", 80, 40, crop=True)
# prepare_png_images("test_stage_1", "adjacent-brain-cropped", 80, 40, crop=True)

