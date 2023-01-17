!pip install imagecodecs
!pip install glob3
from pathlib import Path
import multiprocessing
import scipy
import os
from PIL import Image
import numpy as np
import skimage.io
import tqdm
import glob
import cv2

Image.MAX_IMAGE_PIXELS = None
def crop_white(image, mask, value=255):
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image, mask
    new_image = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    new_mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return new_image, new_mask


def to_jpeg(mask_path):
    path = os.path.join(
        '../input/prostate-cancer-grade-assessment/train_images',
        os.path.basename(mask_path)[:-10] + '.tiff'
    )
    image = skimage.io.MultiImage(path)
    image_mask = skimage.io.MultiImage(mask_path)
    image_to_jpeg(mask_path[:-5] + '_1', image[1], image_mask[1][:,:,0])
    image_to_jpeg(mask_path[:-5] + '_2', image[2], image_mask[2][:,:,0])


def image_to_jpeg(mask_path, image, image_mask):
    image, image_mask = crop_white(image, image_mask)
    image_mask = scipy.sparse.csc_matrix(image_mask)
    scipy.sparse.save_npz(os.path.join('image_masks/', os.path.basename(mask_path)), image_mask)
os.makedirs('image_masks', exist_ok=True)
paths = glob.glob('../input/prostate-cancer-grade-assessment/train_label_masks/*.tiff')
with multiprocessing.Pool(processes=4) as pool:
    for _ in tqdm.tqdm(pool.imap(to_jpeg, paths), total=len(paths)):
        pass
import tarfile
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        
make_tarfile('image_masks.tar.gz','image_masks')
import shutil
shutil.rmtree('image_masks')