import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# put your path to your images
original_path = '../input/dataset/dataset/15-24-500-500-100/'

basepath = original_path + 'mask_annotations/'
ext = ".png"

orig_filename = basepath.split("/")[-3]
orig_filename

saved_rib_mask_filename = original_path + orig_filename + '_rib_mask' + ext
saved_rib_mask_filename
# put all filenames in a list
from glob import glob
mask_all = []
mask_current = []

for path in sorted(glob(basepath + "*")):
    mask_current = path.split("/")[-1]
    #print(mask_current)
    mask_all = sorted(glob(basepath + "*"))
len(mask_all)
# exclude heart, lung, clavicle bones
exclude_masks = ['heart', 'lung', 'clavicle']
rib_mask_all = [word for word in mask_all if not any(bad in word for bad in exclude_masks)]
len(rib_mask_all)
# now compute the union image
unionMask = plt.imread(rib_mask_all[1])
for i in range(2,len(rib_mask_all)):
    add_img = plt.imread(rib_mask_all[i])
    unionMask = cv2.bitwise_or(unionMask, add_img)

cv2.imwrite(orig_filename + '_rib_mask' + ext, (unionMask * 255).astype(np.uint8))
cv2.imwrite(saved_rib_mask_filename, (unionMask * 255).astype(np.uint8))
fig = plt.figure(figsize=(16,10))
plt.imshow(unionMask,cmap='gray')
plt.title('OR image for all ribs')
#plt.close(fig) 
