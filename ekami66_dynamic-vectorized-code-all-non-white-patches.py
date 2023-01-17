import numpy as np

import pandas as pd

import glob

import gc

from pathlib import Path

from openslide import OpenSlide

from tqdm import tqdm
TRAIN_DIR = Path('/kaggle/input/prostate-cancer-grade-assessment/train_images/')

wsis_subset = glob.glob(f'{str(TRAIN_DIR)}/*.tiff')[:10]

patch_size = 128
def extract_non_white_positions(slide_file, patch_size, cuts=2):

    """

        Method used to extract the non white patch positions of a WSI, ignore the patch_size margins at the right and bottom of the WSI

    :param slide_file: The slide file

    :param patch_size: patch size

    :param cuts: Lower number = faster process but require more RAM/Higher number = slower but require less RAM. 

        Play with that setting if you get an OOM on the Kaggle kernel on the test set. 

        If you train on your own machine and you have enough RAM you probably want to set that parameter to 1

    :return: Patches positions

    """

    slide_img = OpenSlide(str(slide_file))

    slide_width, slide_height = slide_img.dimensions



    # Make sure x_cuts is a multiple of patch_size

    x_cuts = np.arange(0, slide_width, step=patch_size)

    x_cuts = [x_cuts[i] for i in range(0, len(x_cuts), len(x_cuts) // cuts)]



    # The patch_size on the right and at the bottom of the WSI will be cropped out, not a big deal

    x_cuts.append(slide_width)

    patches_pos = None

    for cut_idx in range(1, cuts + 1):  # Horizontal sliding window

        start = (x_cuts[cut_idx - 1], 0)

        stop_size = (x_cuts[cut_idx] - x_cuts[cut_idx - 1], slide_height)

        # Always open at lvl 0 because of a bug with Openslide with white patches and always using lvl 0 pos

        pil_img = slide_img.read_region(start, 0, stop_size)

        img = np.array(pil_img)



        # If there is only 1 color (we assume it's either all black or all white)

        if len(set(np.array(pil_img.getextrema()).flatten())) <= 1:

            continue



        imsize = (img.shape[0], img.shape[1])

        nx, ny = (int(dim / patch_size) for dim in imsize)

        img = img[:nx * patch_size, :ny * patch_size, :]



        # reshape padded image according to patches; doesn't copy memory

        patched = img.reshape(nx, patch_size, ny, patch_size, img.shape[2]).transpose(0, 2, 1, 3, 4)

        # check for white patches

        filt = ~(patched == 255).all((2, 3, 4))

        patch_x, patch_y = filt.nonzero()  # patch indices of non-whites from 0 to nx-1, 0 to ny-1

        patch_pixel_x = patch_x * patch_size  # proper pixel indices of each pixel

        patch_pixel_y = patch_y * patch_size

        transposed_pos = np.array([patch_pixel_y, patch_pixel_x]).T

        if patches_pos is None:

            patches_pos = transposed_pos

        else:

            patches_pos = np.concatenate([patches_pos, transposed_pos])

        del img

        gc.collect()

    return patches_pos
wsi_patches = {}

for wsi in tqdm(wsis_subset, desc="Filtering out white patches"):

    wsi_file = TRAIN_DIR / wsi

    patch_pos = extract_non_white_positions(wsi_file, patch_size, cuts=4)

    wsi_patches[wsi] = patch_pos

    print(f"Size of {wsi} patches = {len(patch_pos)}")

    

print(f"Done! Do something with your patches :)")