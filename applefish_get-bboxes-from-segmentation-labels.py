import numpy as np

from skimage.measure import label, regionprops



def rle_decode(mask_rle, shape=(768, 768)):

    """

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    """

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape((shape[1], shape[0])).T  # Needed to align to RLE direction



def masks_as_image(in_mask_list, all_masks=None, shape=(256, 1600)):

    

    # Take the individual masks and create a single mask array

    if all_masks is None:

        all_masks = np.zeros(shape, dtype = np.int16)

    #if isinstance(in_mask_list, list):

    for mask in in_mask_list:

        if isinstance(mask, str):

            all_masks += rle_decode(mask, shape)

    return np.expand_dims(all_masks, -1)



def get_bboxes_from_rle(encoded_pixels, return_mask=False):

    """get all bboxes from a whole mask label"""

    

    mask = masks_as_image([encoded_pixels])

    lbl = label(mask) 

    

    props = regionprops(lbl)



    #get bboxes by a for loop

    bboxes = []

    for prop in props:

        bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[4], prop.bbox[3]])



    if return_mask:

        return bboxes, mask 

    return bboxes
import cv2

import pandas as pd

from skimage.io import imread

import matplotlib.pyplot as plt



ann_file_path = '../input/severstal-steel-defect-detection/train.csv'

train_image_dir = '../input/severstal-steel-defect-detection/train_images/'

images = ['0002cc93b.jpg', '0007a71bf.jpg', '000789191.jpg'] # some images for test



ann_csv = pd.read_csv(ann_file_path)



for image in images:



    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))

    img_0 = cv2.imread(train_image_dir+'/' + image)

    img_1 = img_0.copy()

    print ('Image', image)

    

    # a loop of 4 classes

    for i in range(4):

        rle_0 = ann_csv.query('ImageId_ClassId=="'+image+'_'+str(i+1)+'"')['EncodedPixels'].values[0]

        if isinstance(rle_0, str)!=True and i!=0:

            continue

        bboxes, mask_0 = get_bboxes_from_rle(rle_0, True)

    

        if i == 0:

            color = (255, 0, 0)                

        elif i == 1:

            color = (0, 255, 0)

        elif i == 2:

            color = (0, 0, 255)

        else:

            color = (0, 0, 0)



        for bbox in bboxes:

            print('Found bbox', bbox)

            cv2.rectangle(img_1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color,5)

             

    ax1.imshow(img_0)

    ax1.set_title('Image')

    ax2.set_title('Mask')

    ax3.set_title('Image with derived bounding box')

    ax2.imshow(mask_0[...,0], cmap='gray')

    ax3.imshow(img_1)

    plt.show()
import pandas as pd

import pickle

from tqdm import tqdm



Ids = ann_csv.loc[:, 'ImageId_ClassId'].values

label_category = 4



output = []

for i in tqdm(range(len(Ids)//label_category)):

    # generate anns of all samples into a json file

    

    masks = []

    bboxes = []

    labels = []

    

    for j in range(label_category):

        encoded_pixels = ann_csv.loc[i+j:i+j+1, 'EncodedPixels'].values[0]

        current_bboxes = get_bboxes_from_rle(encoded_pixels)

        current_labels = list(np.ones([len(current_bboxes)]) * (j+1))

        

        bboxes.extend(current_bboxes)

        labels.extend(current_labels)

        masks.append({

            u'counts':encoded_pixels,

            u'size':[256, 1600]

        })

        

    bboxes = np.array(bboxes)

    labels = np.array(labels, dtype=int)

    masks = np.array(masks)

    

    dict0 = {

        'filename': Ids[i][:-2],

        'width': 1600,

        'height': 256,

        'ann': 

        {

            'bboxes': bboxes, # shape is (sample_num, bbox_num)

            'labels': labels, # shape is (sample_num, bbox_num)

            'masks': masks  # shape is (sample_num, 4)

        }

        }

    output.append(dict0)



# save output as a pickle file

with open('train_anns.pkl', 'wb') as f:

    pickle.dump(output, f)
# load the file

with open("train_anns.pkl", "rb") as f:

    anns = pickle.load(f)



# print some labels    

print('frist label:')

print(anns[0])

print('')

print('second label:')

print(anns[1])

print('')

print('third label:')

print(anns[2])