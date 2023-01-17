import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import seaborn as sns



from PIL import Image



import os



%matplotlib inline



print(os.listdir('../input/'))
train_labels = pd.read_csv('../input/understanding_cloud_organization/train.csv')

train_labels.head()
len(train_labels)
# Seperating image and label:

train_labels['label'] = train_labels['Image_Label'].apply(lambda x: x.split('_')[1])

train_labels['image'] = train_labels['Image_Label'].apply(lambda x: x.split('_')[0])
# Changind all the NaNs by -1

train_labels['EncodedPixels'] = train_labels['EncodedPixels'].fillna(-1)



# Remvoing all the -1 and only keeping the ones with actual Encoded Pixels

train_no_nans = train_labels[train_labels.EncodedPixels != -1]

train_no_nans.head()
labels = ['Fish', 'Flower', 'Gravel', 'Sugar']

nb_labels = []

for item in labels:

    nb_label = train_no_nans['label'].str.count(item)

    nb_labels.append(nb_label[nb_label == 1].count())
# Number of image containing each label



train_no_nans['label'].value_counts().plot(kind='bar')
images = train_no_nans['image'].unique()

images[:10]
# Printing the first few images and there labels



for img_nb in range(2):

    print(train_no_nans.loc[train_labels['image'] == images[img_nb]]['label'])

    img = plt.imread('../input/understanding_cloud_organization/train_images/' + images[img_nb])

    plt.imshow(img)

    plt.show()
#%%time

# Creating masks



def making_mask(nb_img_to_mask):

    

    """

    nb_img_to_mask - number of images to be converted to mask.

    """

#     print("nb_img_to_mask" , nb_img_to_mask)

    all_masks = []

    row=0

    for index in range(nb_img_to_mask):

#         print('\nfor image: ', index)

        # Creating a 4d array (1d for each label)

        mask = np.zeros((1400, 2100, 4))

        

        # Select 4 rows:

        # Why selecting only 4 rows? 

        # Bcoz, there are 4 rows for each label. 

        labels = train_labels.iloc[row:row+4]

        band=0

    

        for label in labels.values:

            

            # If there is a mask for a given label

            if label[1] != -1:

                list_pixel = label[1].split()  

                # Create an mask the size of the image, with only zeros

                mask_label = np.zeros((1400,2100))

                # Store position of 1st pixel & length of string

                positions = list_pixel[::2] # ["264918" , "266318" , ...]

                length = list_pixel[1::2] # ["985" , "956" , ....]

                for pos, le in zip(positions, length):

                    start = int(pos)

                    start_column = start//1400

                    start_row = start%1400

                    end = int(le)

#                     print(start_row, start_row+end,start_column)

                    mask_label[start_row:start_row+end,start_column] = 1

                

                mask[:,:,band] = mask_label

            band+=1

        all_masks.append(mask)

        index +=1

        row +=4

        

    return all_masks
masks_images = making_mask(10)
def plot_mask_on_img(mask, img_nb):

    label_list = ['Fish', 'Flower', 'Gravel', 'Sugar']

    

    img = plt.imread('../input/understanding_cloud_organization/train_images/' + images[img_nb])

    plt.figure(figsize=[30, 10])

    for label in range(4):

        if 1 in mask[:,:,label]:

            print(label_list[label])

            plt.subplot(1,4,label+1)

            plt.imshow(img)

            plt.imshow(mask[:,:,label], alpha=0.3, label=label_list[label])

            

    plt.show()
plot_mask_on_img(masks_images[0], 0)
plot_mask_on_img(masks_images[1], 1)
%matplotlib inline

import numpy as np

import cv2

from matplotlib import pyplot as plt



from albumentations import (

    PadIfNeeded,

    HorizontalFlip,

    VerticalFlip,    

    CenterCrop,    

    Crop,

    Compose,

    Transpose,

    RandomRotate90,

    ElasticTransform,

    GridDistortion, 

    OpticalDistortion,

    RandomSizedCrop,

    OneOf,

    CLAHE,

    RandomBrightnessContrast,    

    RandomGamma    

)
def visualize(image, mask, original_image=None, original_mask=None):

    fontsize = 18

    

    if original_image is None and original_mask is None:

        f, ax = plt.subplots(2, 1, figsize=(8, 8))



        ax[0].imshow(image)

        ax[1].imshow(mask)

    else:

        f, ax = plt.subplots(2, 2, figsize=(8, 8))



        ax[0, 0].imshow(original_image)

        ax[0, 0].set_title('Original image', fontsize=fontsize)

        

        ax[1, 0].imshow(original_mask)

        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        

        ax[0, 1].imshow(image)

        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        

        ax[1, 1].imshow(mask)

        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
image = plt.imread('../input/understanding_cloud_organization/train_images/' + images[0])

mask = masks_images[0][:,:,0] # Fish
aug = PadIfNeeded(p=1, min_height=128, min_width=128)



augmented = aug(image=image, mask=mask)



image_padded = augmented['image']

mask_padded = augmented['mask']



print(image_padded.shape, mask_padded.shape)



visualize(image_padded, mask_padded, original_image=image, original_mask=mask)
aug = HorizontalFlip(p=1)



augmented = aug(image=image, mask=mask)



image_h_flipped = augmented['image']

mask_h_flipped = augmented['mask']



visualize(image_h_flipped, mask_h_flipped, original_image=image, original_mask=mask)
aug = VerticalFlip(p=1)



augmented = aug(image=image, mask=mask)



image_v_flipped = augmented['image']

mask_v_flipped = augmented['mask']



visualize(image_v_flipped, mask_v_flipped, original_image=image, original_mask=mask)
aug = RandomRotate90(p=1)



augmented = aug(image=image, mask=mask)



image_rot90 = augmented['image']

mask_rot90 = augmented['mask']



visualize(image_rot90, mask_rot90, original_image=image, original_mask=mask)
aug = Transpose(p=1)



augmented = aug(image=image, mask=mask)



image_transposed = augmented['image']

mask_transposed = augmented['mask']



visualize(image_transposed, mask_transposed, original_image=image, original_mask=mask)


aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)



augmented = aug(image=image, mask=mask)



image_elastic = augmented['image']

mask_elastic = augmented['mask']



visualize(image_elastic, mask_elastic, original_image=image, original_mask=mask)
aug = GridDistortion(p=1)



augmented = aug(image=image, mask=mask)



image_grid = augmented['image']

mask_grid = augmented['mask']



visualize(image_grid, mask_grid, original_image=image, original_mask=mask)
aug = Transpose(p=1)



augmented = aug(image=image, mask=mask)



image_transposed = augmented['image']

mask_transposed = augmented['mask']



visualize(image_transposed, mask_transposed, original_image=image, original_mask=mask)