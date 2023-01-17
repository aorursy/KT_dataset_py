import cv2

import matplotlib

import pandas as pd

import numpy as np

from tqdm import tqdm

from PIL import Image

import random





train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')



train_data = np.load('/kaggle/input/numpy256siim/x_train_256.npy')

import os



hairs = []



if not os.path.exists('/kaggle/working/hairs/'):

    os.makedirs('/kaggle/working/hairs/')





for i in tqdm(range(len(train_df))):

    image_id = train_df.iloc[i]['image_name']

    image = train_data[i]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





    src = cv2.resize(image, (256, 256))



    # Convert the original image to grayscale

    grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )



    # Kernel for the morphological filtering

    kernel = cv2.getStructuringElement(1,(17,17))



    # Perform the blackHat filtering on the grayscale image to find the 

    # hair countours

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    ret,thresh2 = cv2.threshold(blackhat,50,255,cv2.THRESH_BINARY)

    if (thresh2.sum()/255) > 1000:

#         print( thresh2.sum()/255 )

        hair = cv2.bitwise_and(src, src, mask=thresh2)

        hairs.append(hair)

        cv2.imwrite(f'/kaggle/working/hairs/{image_id}.png', hair)

import matplotlib.pyplot as plt





for i in range(9):

    plt.subplot(3, 3, i+1)

    train_image_index = np.random.randint(0, len(train_data))

    hair_image_index = np.random.randint(0, len(hairs))

    image = train_data[train_image_index]

    hair = hairs[hair_image_index]

    # loop over the image, pixel by pixel

    image[hair>=1] = 0

    dst = cv2.add(image, hair)

    plt.imshow(dst)

    plt.axis('off')
class HairAugmentation:

    def __init__(self, hairs_folder="/kaggle/working/hairs/"):

        self.hairs_folder = hairs_folder



    def __call__(self, img):

        

        

        n_hair = random.randint(0, len(os.listdir(self.hairs_folder))-1)

        hair = os.listdir(self.hairs_folder)[n_hair]      

        hair_image = cv2.imread(os.path.join(self.hairs_folder, hair))



        img[hair_image>=1] = 0



        img = cv2.add(img, hair_image)

            

        return img

    

    def __repr__(self):

        return f'{self.__class__.__name__}(hairs_folder="{self.hairs_folder}")'
hair_augmentation = HairAugmentation()
if not os.path.exists('/kaggle/working/hair_augmentations/'):

    os.makedirs('/kaggle/working/hair_augmentations/')





for i in tqdm(range(len(train_data))):

    image_id = train_df.iloc[i]['image_name']

    image = hair_augmentation(train_data[i])

    cv2.imwrite(f'/kaggle/working/hair_augmentations/{image_id}.jpg', image)