import os
import numpy as np
from skimage.io import imread

TRAIN_PATH = "../input/stage1_train/"

for image in os.listdir(TRAIN_PATH):
    mask_list = os.listdir("%s%s/masks" % (TRAIN_PATH, image))
    for mask in mask_list:
        full_path = "%s%s/masks/%s" % (TRAIN_PATH, image, mask)
        inp_mask = imread(full_path)
        if (np.count_nonzero(inp_mask) != 0):
            x_mask, y_mask = np.nonzero(inp_mask)
            size_x = max(x_mask) - min(x_mask) + 1
            size_y = max(y_mask) - min(y_mask) + 1
            if size_x < 2 or size_y < 2: # Flag Mask
                print(full_path)
        else:
            print("Empty mask: %s" % (full_path))