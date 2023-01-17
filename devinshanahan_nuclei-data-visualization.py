# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_labels = pd.read_csv("../input/stage1_train_labels.csv")
train_labels.head()
# number of unique ImageIds

images = train_labels['ImageId'].unique()

print(len(images))
def img_by_row(row):

    image_id = train_labels['ImageId'].iloc[row]

    image = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)

    image = cv2.imread(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image



def img_by_id(image_id):

    image = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)

    image = cv2.imread(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
def random_train_images(df):

    random_images = np.random.permutation(df['ImageId'])[:36]

    

    f, axarr = plt.subplots(6,6,figsize=(16,11))

    for row in range(6):

        for col in range(6):

            image_id = random_images[row*6 + col]

            image = img_by_id(image_id)

            ax = axarr[row,col]

            ax.axis('off')

            ax.set_title(str(image.shape))

            ax.imshow(image)
# display a random set of images with image sizes

random_train_images(train_labels)
def mask_by_id(image_id):

    df = train_labels[train_labels['ImageId'] == image_id]

    filled_pixels = []

    for img in range(len(df)):

        encoded = list(map(int,df['EncodedPixels'].iloc[img].split()))

        for i in range(0,len(encoded),2):

            filled_pixels.append(encoded[i])

            for j in range(1,encoded[i+1] + 1):

                filled_pixels.append(encoded[i] + j)

    image = img_by_id(image_id)

    h = image.shape[0]

    w = image.shape[1]

    mask = np.zeros((h*w,1))

    for px in filled_pixels:

        mask[px - 2] = 1

    mask = mask.reshape((h,w),order='F')

    return mask
def display_image_masks(image_id):

    image = img_by_id(image_id)

    mask = mask_by_id(image_id)

    f = plt.figure(figsize=(10,20))

    ax1 = f.add_subplot(121)

    ax1.axis('off')

    ax1.imshow(image)

    ax1.set_title("original")

    ax2 = f.add_subplot(122)

    ax2.axis('off')

    ax2.imshow(mask)

    ax2.set_title("masks")
r = np.random.randint(0,len(images))

print("ImageId:",images[r])

display_image_masks(images[r])