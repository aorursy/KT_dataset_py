import os

import numpy as np

import pandas as pd

import warnings

import cv2

import random

from matplotlib import pyplot as plt



%matplotlib inline

warnings.filterwarnings("ignore")
print(os.getcwd())

print(os.listdir('../input'))
def show(img):

#     plt.figure()

    plt.imshow(img)

#     plt.show()
def rle_to_mask(rle_string, img):

    

    rows, cols = img.shape[0], img.shape[1]

    img = np.zeros(rows*cols, dtype=np.uint8)



    rle_numbers = [int(x) for x in rle_string.split(' ')]

    rle_pairs = np.array(rle_numbers).reshape(-1,2)



    for index, length in rle_pairs:

        index -= 1

        img[index:index+length] = 255

    img = img.reshape(cols,rows)

    img = img.T

    img = image = np.expand_dims(img, axis=2)

    

    return img
def ignore_background(img_mask, img_origin):

    assert img_mask.shape == img_mask.shape

    

    result = img_mask.copy()

    result[np.where(img_mask==255)] = img_origin[np.where(img_mask==255)]

    

    return result
train_csv = pd.read_csv('../input/understanding_cloud_organization/train.csv')

sub_csv = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')
train_csv = train_csv.fillna(-1)

train_csv.head()
train_csv['ImageId'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[0])

train_csv['Label'] = train_csv['Image_Label'].apply(lambda x: x.split('_')[1])

train_csv = train_csv.drop('Image_Label', axis=1)

train_csv.head()
print('Total', len(train_csv['ImageId'].unique()),'Images for', len(train_csv['Label'].unique()), 'Types.')
fish = len(train_csv[train_csv['Label']=='Fish'][train_csv['EncodedPixels']!= -1])

flower = len(train_csv[train_csv['Label']=='Flower'][train_csv['EncodedPixels']!= -1])

gravel = len(train_csv[train_csv['Label']=='Gravel'][train_csv['EncodedPixels']!= -1])

sugar = len(train_csv[train_csv['Label']=='Sugar'][train_csv['EncodedPixels']!= -1])

print('The amount of Fish:{0}, Flower:{1}, Gravel:{2}, Sugar:{3} .'.format(fish, flower, gravel, sugar))

print('Totally {} valid Images.'.format(fish+flower+gravel+sugar))
plt.title("Total amount of images each type.")

plt.bar([1,2,3,4],[fish, flower, gravel, sugar], tick_label=['Fish', 'Flower', 'Graver', 'Sugar'])
train_df = train_csv[train_csv['EncodedPixels']!=-1]

train_df.shape
types_per_image = train_df.groupby(by='ImageId', as_index=False).agg({'EncodedPixels': pd.Series.nunique})['EncodedPixels']
per = np.histogram(list(types_per_image), bins=range(1, 6))
plt.title("Histogram of types per image.")

plt.bar([1,2,3,4], per[0])
BASE_DIR = '../input/understanding_cloud_organization/train_images/'
train_df.head()
img = cv2.imread(BASE_DIR + train_df['ImageId'][7])

img = cv2.resize(img, (512, 512))
show(img);plt.axis('off')
horizontal_img = cv2.flip( img, 0 )

vertical_img = cv2.flip( img, 1 )

both_img = cv2.flip( img, -1 )
(h, w) = img.shape[:2] #10

center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, 40, 1.0) #12

rotated = cv2.warpAffine(img, M, (w, h))


lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)



lab_planes = cv2.split(lab)



clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))



lab_planes[0] = clahe.apply(lab_planes[0])



lab = cv2.merge(lab_planes)



bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
plt.imshow(bgr);plt.axis('off')
img = cv2.imread(BASE_DIR + train_df['ImageId'][0])

img_mask = rle_to_mask(train_df['EncodedPixels'][0], img)

img_new0 = ignore_background(img_mask, img)

img_mask = rle_to_mask(train_df['EncodedPixels'][1], img)

img_new1 = ignore_background(img_mask, img)
img_mask.shape
plt.subplots(figsize=(15, 6))

plt.subplot(1,3,1); show(img);plt.title('orginal image');

plt.subplot(1,3,2); show(img_new0);plt.title('Fish part');

plt.subplot(1,3,3); show(img_new1);plt.title('Flower part');
show(img)
train_df = train_df.reset_index(drop=True)

for i in range(100):

    index = random.randint(0, 11835)

    img = cv2.imread(BASE_DIR + train_df['ImageId'][index])

    plt.subplots(figsize=(15, 6))

    plt.subplot(1, 3, 1); show(img);

    plt.title('Origin Image. Index: {} Type: {}'.format(index, train_df['Label'][index]))

    plt.subplot(1, 3, 2); show(rle_to_mask(train_df['EncodedPixels'][index], img));

    plt.title('Mask. Index: {} Type: {}'.format(index, train_df['Label'][index]))

    plt.subplot(1, 3, 3); show(ignore_background(rle_to_mask(train_df['EncodedPixels'][index], img), img));

    plt.title('Masked Image. Index: {} Type: {}'.format(index, train_df['Label'][index]))
train_df.head()
image_id = list(train_df['ImageId'].unique())
warm = []

for x in image_id:

    img = cv2.imread(BASE_DIR + x)

    tmp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    types = list(train_df[train_df['ImageId']==x]['EncodedPixels'])

    for y in types:

        print(rle_to_mask(y, img).shape)

        tmp = tmp + rle_to_mask(y, img)/255.0

    warm.append(np.max(tmp))
plt.title('Amount of Types per Pixel has')

plt.bar([1,2,3,4], np.histogram(warm, bins=range(1, 6))[0])
import pandas as pd

ensemble = pd.read_csv("../input/ensemble/ensemble.csv")
ensemble.head(30)
result = cv2.imread('../input/understanding_cloud_organization/test_images/969f34b.jpg')

result = cv2.resize(result, (525, 350))
show(result)
img_mask = rle_to_mask(ensemble['EncodedPixels'][1], result)

img_new0 = ignore_background(img_mask, result)

img_mask = rle_to_mask(ensemble['EncodedPixels'][3], result)

img_new1 = ignore_background(img_mask, result)
plt.subplots(figsize=(15, 6))

plt.subplot(1,3,1); show(result);plt.title('orginal image');

plt.subplot(1,3,2); show(img_new0);plt.title('Flower part');

plt.subplot(1,3,3); show(img_new1);plt.title('Sugar part');
result = cv2.imread('../input/understanding_cloud_organization/test_images/5a61caf.jpg')

result = cv2.resize(result, (525, 350))
show(result)
img_mask = rle_to_mask(ensemble['EncodedPixels'][21], result)

img_new0 = ignore_background(img_mask, result)

img_mask = rle_to_mask(ensemble['EncodedPixels'][22], result)

img_new1 = ignore_background(img_mask, result)

img_mask = rle_to_mask(ensemble['EncodedPixels'][23], result)

img_new2 = ignore_background(img_mask, result)
plt.subplots(figsize=(15, 6))

plt.subplot(1,4,1); show(result);plt.title('orginal image');

plt.subplot(1,4,2); show(img_new0);plt.title('Flower part');

plt.subplot(1,4,3); show(img_new1);plt.title('Gravel part');

plt.subplot(1,4,4); show(img_new2);plt.title('Sugar part');