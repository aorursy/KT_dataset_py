opts = {}

opts['imageType_train'] = '.tif'

opts['imageType_test'] = '.tif'

opts['number_of_channel'] = 3                   # Set if to '3' for RGB images and set it to '1' for grayscale images

opts['treshold'] = 0.5                          # treshold to convert the network output (stage 1) to binary masks

## input & output directories

opts['train_dir'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/tissue images/'

opts['train_label_dir'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/mask binary/'

opts['train_label_dir_modify'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/mask binary without border/'

opts['train_label_masks'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/label masks modify/'

opts['train_dis_dir'] = '../input/segmentation-of-nuclei-in-cryosectioned-he-images/distance maps/'

opts['results_save_path'] ='/kaggle/working/images/'

opts['models_save_path'] ='/kaggle/working/models/'

import numpy as np

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import matplotlib.pyplot as plt

from glob import glob                                 # path control

import tqdm

from skimage.io import imsave, imread

from skimage.morphology import label

train_files = glob('{}*{}'.format(opts['train_dir'], opts['imageType_train']))

train_files_mask = glob('{}*.png'.format(opts['train_label_dir']))

train_files_mask_modify = glob('{}*.png'.format(opts['train_label_dir_modify']))

train_files_dis = glob('{}*.png'.format(opts['train_dis_dir']))

train_files_labels = glob('{}*.tif'.format(opts['train_label_masks']))





train_files.sort()

train_files_mask.sort()

train_files_dis.sort()

train_files_labels.sort()

train_files_mask_modify.sort()

print("Total number of training images:", len(train_files))
# we have 10 organ in this dataset

train_files
# an exmaple for Adrenal Gland image

id = 0

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw adrenal gland image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Larynx image

id = 4

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw larynx image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Lymph Nodes image

id = 7

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw lymph nodes image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Mediastinum image

id = 10

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw mediastinum image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Pancreas image

id = 13

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw pancreas image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Pleura image

id = 16

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw pleura image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Skin image

id = 19

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw skin image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Testes image

id = 22

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw testes image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Thymus image

id = 25

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw thymus image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# an exmaple for Thyroid Gland image

id = 28

plt.figure(figsize=(30,60))

plt.subplot(1,5,1)

plt.imshow(imread(train_files[id]), cmap = 'gray')

plt.axis('off') 

plt.title('raw thyroid gland image', fontsize = 20)



plt.subplot(1,5,2)

plt.imshow(imread(train_files_mask[id]))

plt.axis('off')

plt.title('binary mask', fontsize = 20)



plt.subplot(1,5,3)

plt.imshow(imread(train_files_mask_modify[id]))

plt.axis('off')

plt.title('binary mask (boder removed)', fontsize = 20)



plt.subplot(1,5,4)

plt.imshow(imread(train_files_dis[id]))

plt.axis('off')

plt.title('distance map', fontsize = 20)



plt.subplot(1,5,5)

plt.imshow(imread(train_files_labels[id]))

plt.axis('off')

plt.title('label mask', fontsize = 20)
# number of annotated nuclie

count = []

for i in range(len(train_files_labels)):

    img = imread(train_files_labels[i])

    count.append(len(np.unique(img)))



Adrenal_Gland = np.sum(count[0:3])

Larynx        = np.sum(count[3:6])

Lymph_Nodes   = np.sum(count[6:9])

Mediastinum   = np.sum(count[9:12])

Pancreas      = np.sum(count[12:15])

Pleura        = np.sum(count[15:18])

Skin          = np.sum(count[18:21])

Testes        = np.sum(count[21:24])

Thymus        = np.sum(count[24:27])

Thyroid_Gland = np.sum(count[27:30])



print('total number of annotated nuclei is:', np.sum(count))    

print('==============================================')

print('total number of annotated adrenal gland nuclei is:', Adrenal_Gland)    

print('total number of annotated larynx nuclei is:', Larynx)    

print('total number of annotated lymph nodes nuclei is:',Lymph_Nodes)     

print('total number of annotated mediastinum nuclei is:', Mediastinum)    

print('total number of annotated pancreas nuclei is:', Pancreas)    

print('total number of annotated pleura nuclei is:', Pleura)    

print('total number of annotated skin nuclei is:', Skin)    

print('total number of annotated testes nuclei is:', Testes)    

print('total number of annotated thymus nuclei is:', Thymus)    

print('total number of annotated thyroid gland nuclei is:', Thyroid_Gland)    


