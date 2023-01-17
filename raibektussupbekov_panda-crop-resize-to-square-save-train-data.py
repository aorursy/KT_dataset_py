import numpy as np

import cv2

from pathlib import Path

from skimage.io import MultiImage

import pandas as pd

import matplotlib.pyplot as plt

from multiprocessing import Pool

from tqdm.notebook import tqdm



def crop_white(image: np.ndarray, value: int = 255) -> np.ndarray:

    assert image.shape[2] == 3

    assert image.dtype == np.uint8

    ys, = (image.min((1, 2)) < value).nonzero()

    xs, = (image.min((0, 2)) < value).nonzero()

    

    # if there's no pixel with such a value

    if len(xs) == 0 or len(ys) == 0:

        return image

    

    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]



def resize_to_square(image: np.ndarray, img_size: int = 224, color: list = [255, 255, 255]) -> np.ndarray:

    old_size = image.shape[:2] # old_size is in (height, width) format

    ratio = float(img_size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = img_size - new_size[1]

    delta_h = img_size - new_size[0]

    top, bottom = delta_h//2, delta_h-(delta_h//2)

    left, right = delta_w//2, delta_w-(delta_w//2)

    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_image



def process_tiff(tiff_path: Path) -> bool:

    multi_image = MultiImage(str(tiff_path))

    assert cv2.imwrite(

        str(SAVE_PATH.joinpath(tiff_path.with_suffix('.png').name)),

        cv2.cvtColor(resize_to_square(crop_white(multi_image[TIFF_LEVEL], PIXEL_VALUE), IMG_SIZE, [PIXEL_VALUE, PIXEL_VALUE, PIXEL_VALUE]), cv2.COLOR_RGB2BGR)

    )



TIFF_LEVEL = 1

IMG_SIZE = 224

PIXEL_VALUE = 255

TRAIN_IMAGES_PATH = Path("../input/prostate-cancer-grade-assessment/train_images/")

SAVE_PATH = Path("train_images")



SAVE_PATH.mkdir(exist_ok=True)
img = MultiImage('../input/prostate-cancer-grade-assessment/train_images/00412139e6b04d1e1cee8421f38f6e90.tiff')



fig, ax = plt.subplots(1, 3, figsize=(16, 8))



ax[0].imshow(img[TIFF_LEVEL])

ax[0].set_title(f'Original, shape={img[TIFF_LEVEL].shape}')



img_cropped = crop_white(img[TIFF_LEVEL], PIXEL_VALUE) 

ax[1].imshow(img_cropped)

ax[1].set_title(f'Cropped, shape={img_cropped.shape}')



img_resized = resize_to_square(img_cropped, IMG_SIZE, [PIXEL_VALUE, PIXEL_VALUE, PIXEL_VALUE]) 

ax[2].imshow(img_resized)

ax[2].set_title(f'Resized, shape={img_resized.shape}')



plt.show()
import os



print(f'{os.cpu_count()} CPU available')
train_images = list(TRAIN_IMAGES_PATH.glob('*.tiff'))



with Pool() as p:

    with tqdm(total=len(train_images)) as pbar:

        for r in p.imap_unordered(process_tiff, train_images):

            pbar.update()
!zip -r -q train_images.zip train_images