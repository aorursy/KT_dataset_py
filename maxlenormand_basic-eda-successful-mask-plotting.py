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
# Creating masks



def making_mask(nb_img_to_mask):

    all_masks = []

    complex_shapes = 0

    rectangles = 0

    row=0

    for index in range(nb_img_to_mask):

        #print('\nfor image: ', index)

        # Creating a 4d array (1d for each label)

        mask = np.zeros((1400, 2100, 4))

        

        # Select 4 rows:

        labels = train_labels.iloc[row:row+4]

        band=0

        for label in labels.values:

            

            # If there is a mask for a given label

            if label[1] != -1:

                

                list_pixel = label[1].split()

                # Create an mask the size of the image, with only zeros

                mask_label = np.zeros((1400,2100))

                # Store position of 1st pixel & length of string

                positions = list_pixel[::2]

                length = list_pixel[1::2]

                

                # If the length is always the same (we'll assume then there's only a rectangle)

                unique_values = np.unique(length)

                

                if len(unique_values)==1:

                    #print('rectangle shape')

                    rectangles+=1

                    #We make a rectangle starting from the top left, so 1st pixel in postions

                    start = int(positions[0])

                    start_column = start//1400

                    start_row = start%1400

                    end = int(length[0])

                    # Create the masks that starts on top left, width nb elements in length[] &

                    # length nb elements in positions

                    mask_label[start_row:start_row+end,start_column:start_column+len(positions)] = 1

                    

                else:

                    #print('complex shape')

                    complex_shapes+=1

                    for pos, le in zip(positions, length):

                        start = int(pos)

                        start_column = start//1400

                        start_row = start%1400

                        end = int(le)

                        mask_label[start_row:start_row+end,start_column] = 1

                

                mask[:,:,band] = mask_label

            band+=1

        all_masks.append(mask)

        index +=1

        row +=4

        

    return all_masks, complex_shapes, rectangles
masks_images, nb_complex, nb_rectangles = making_mask(100)

print(f'There were:\n{nb_complex} complex shapes ({np.round(nb_complex*100/(nb_complex+nb_rectangles),2)}%)\n{nb_rectangles} rectangles ({np.round(nb_rectangles*100/(nb_complex+nb_rectangles),2)}%)')
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
plot_mask_on_img(masks_images[2], 2)