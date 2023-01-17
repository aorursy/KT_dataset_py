import numpy as np
import os
from os.path import splitext
import shutil
import itertools

folders = next(os.walk("."))[1]
paths = [p for p in folders if p.startswith("Slide")]
img_paths = []
for p in paths:
    pw = next(os.walk(p))
    img_paths.extend([[pw[0],pwl] for pwl in pw[2]])
print(img_paths)
paths = [p for p in folders if p.startswith("GT")]
gt_paths = []
for p in paths:
    pw = next(os.walk(p))
    gt_paths.extend([[pw[0],pwl] for pwl in pw[2]])
print(gt_paths)
make_folders = []
make_folders.extend([splitext(f[1])[0] for f in img_paths])
print(make_folders)
for img, gt, f in zip(img_paths, gt_paths, make_folders):
    path_images = os.path.join(f, "images")
    path_masks = os.path.join(f, "masks")
    os.makedirs(path_images)
    os.makedirs(path_masks)
    shutil.copy(os.path.join(img[0], img[1]), path_images)
    shutil.copy(os.path.join(gt[0], gt[1]), path_masks)
from skimage.morphology import label
from skimage.io import imread, imsave

for gt,f in zip(gt_paths, make_folders):
    path_mask = os.path.join(f, "masks", gt[1])
    img = imread(path_mask)
    labels, n = label(img, connectivity=1, return_num=True)
    for i in range(1,n):
        mask = np.where(labels==i, True, False)
        imsave(os.path.join(f, "masks", "mask_{0}.png".format(i)), mask * 255)
    print(n)
# remove original masks
for gt,f in zip(gt_paths, make_folders):
    path_mask = os.path.join(f, "masks", gt[1])
    os.remove(path_mask)