import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

masks_all = []
dir_all = []
input = '../input/maskribs/'
for dir in listdir(input): 
    dir_all.append(dir)
    masks_all.append([input+dir+'/mask_annotations/'+file for file in listdir(input+dir+'/mask_annotations')])

[print(len(subdir)) for subdir in masks_all]
# put all filenames in a list
#from glob import glob
#mask_all = []
#mask_current = []

#for path in sorted(glob(basepath + "*")):
#    mask_current = path.split("/")[-1]
    #print(mask_current)
#    mask_all = sorted(glob(basepath + "*"))
len(mask_all)
# exclude heart, lung, clavicle bones
exclude_masks = ['heart', 'lung', 'clavicle']
#rib_mask_all = [word for word in masks_all if not any(bad in word for bad in exclude_masks)]
rib_masks_all = []
for subdir in masks_all:
     rib_masks_all.append([word for word in subdir if not any(bad in word for bad in exclude_masks)])

for i in range(0,len(masks_all)):
    print('Before:'+str(len(masks_all[i]))+'\tAfter:'+str(len(rib_masks_all[i])))
for i in range(0,len(rib_masks_all)):
    unionMask = plt.imread(rib_masks_all[i][0])
    for j in range(0,len(rib_masks_all[i])):
        add_img = plt.imread(rib_masks_all[i][j])
        unionMask = cv2.bitwise_or(unionMask, add_img)
    cv2.imwrite('../input/'+dir_all[i] + '.png', (unionMask * 255).astype(np.uint8))
    cv2.imwrite(dir_all[i]+'.png', (unionMask * 255).astype(np.uint8))
#print('qweqwe')
fig = plt.figure(figsize=(16,10))
plt.imshow(unionMask,cmap='gray')
plt.title('OR image for all ribs')
plt.close(fig) 
