%matplotlib inline

import numpy as np # linear algebra

import pandas as pd

import cv2

import os

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14, 12]

import collections

from PIL import Image

import matplotlib.image as mpimg

import matplotlib.patches as patches

import random

DIR = "../input/humpback-whale-identification"
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=Warning)
train = pd.read_csv(os.path.join(DIR, "train.csv"))

test = pd.read_csv(os.path.join(DIR, "sample_submission.csv"))

train.shape, test.shape
os.listdir(DIR)
train.head()
test.head()
random_train_whales = np.random.choice([os.path.join(DIR+'/train',whale) for whale in train['Image']],3)

random_test_whales = np.random.choice([os.path.join(DIR+'/test',whale) for whale in test['Image']],3)

both_whales = np.concatenate([random_train_whales,random_test_whales])

print('Training Images:')

for i,whale in enumerate(both_whales):

    if i==3:

        print('Test Images:')

    img = Image.open(whale)

    plt.imshow(img)

    plt.show()
train['Id'].value_counts()[:5]
print(f"There are {len(os.listdir(DIR+'/train'))} images in train dataset with {train.Id.nunique()} unique classes.")

print(f"There are {len(os.listdir(DIR+'/test'))} images in test dataset.")
for i in range(1, 4):

    print(f'There are {train.Id.value_counts()[train.Id.value_counts().values==i].shape[0]} classes with {i} samples in train data.')
plt.title('Distribution of Classes excluding new_whale');

train.Id.value_counts()[1:].plot(kind='hist', bins=8,figsize=(20,14));
counted = train.groupby("Id").count().rename(columns={"Image":"image_count"})

counted.loc[counted["image_count"] > 80,'image_count'] = 80

plt.figure(figsize=(20,14))

sns.countplot(data=counted, x="image_count")

plt.show()
image_count_for_whale = train.groupby("Id", as_index=False).count().rename(columns={"Image":"image_count"})

whale_count_for_image_count = image_count_for_whale.groupby("image_count", as_index=False).count().rename(columns={"Id":"whale_count"})

whale_count_for_image_count['image_total_count'] = whale_count_for_image_count['image_count'] * whale_count_for_image_count['whale_count']
whale_count_for_image_count[:5]
whale_count_for_image_count[-3:]
fig = plt.figure(figsize = (20, 15))

for idx, img_name in enumerate(train[train['Id'] == 'new_whale']['Image'][:12]):

    y = fig.add_subplot(3, 4, idx+1)

    img = cv2.imread(os.path.join(DIR,"train",img_name))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    y.imshow(img)

plt.show()
single_whales = train['Id'].value_counts().index[-12:]

fig = plt.figure(figsize = (20, 15))



for widx, whale in enumerate(single_whales):

    for idx, img_name in enumerate(train[train['Id'] == whale]['Image'][:1]):

        axes = widx + idx + 1

        y = fig.add_subplot(3, 4, axes)

        img = cv2.imread(os.path.join(DIR,"train",img_name))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        y.imshow(img)



plt.show()
def Plot_image_tog(ls,row,col):

    fig = plt.figure(figsize = (20, 15))

    for idx, img_name in enumerate(ls):

        y = fig.add_subplot(row, col, idx+1)

        img = cv2.imread(os.path.join(DIR,"train",img_name))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        y.imshow(img)

    plt.show()
ls = ['0b75361cd.jpg','0c6772887.jpg','0ef9d37be.jpg','fabc19a85.jpg']

Plot_image_tog(ls,2,2)
text_ls=["2b96cac5a.jpg",'f110a9721.jpg','0b6e959b8.jpg','0b7aef92f.jpg','00b92e9bf.jpg','f045d7afc.jpg']

Plot_image_tog(text_ls,2,3)
single_ls=['f3f2023c6.jpg','f0cfd99be.jpg','ed309eb49.jpg','155116572.jpg','0ac7c6cf0.jpg','fdb27aea3.jpg']

#

Plot_image_tog(single_ls,2,3)
# train[train["Image"] == "2b96cac5a.jpg"]

# train[train["Id"] == "w_c7bd8e7"]
imageSizes_train = collections.Counter([Image.open(f'{DIR}/train/{filename}').size

                        for filename in os.listdir(f"{DIR}/train")])

imageSizes_test = collections.Counter([Image.open(f'{DIR}/test/{filename}').size

                        for filename in os.listdir(f"{DIR}/test")])
def isdf(imageSizes):

    imageSizeFrame = pd.DataFrame(list(imageSizes.most_common()),columns = ["imageDim","count"])

    imageSizeFrame['fraction'] = imageSizeFrame['count'] / sum(imageSizes.values())

    imageSizeFrame['count_cum'] = imageSizeFrame['count'].cumsum()

    imageSizeFrame['count_cum_fraction'] = imageSizeFrame['count_cum'] / sum(imageSizes.values())

    return imageSizeFrame



train_isdf = isdf(imageSizes_train)

train_isdf['set'] = 'train'

test_isdf = isdf(imageSizes_test)

test_isdf['set'] = 'test'
isizes = train_isdf.merge(test_isdf, how="outer", on="imageDim")

isizes['total_count'] = isizes['count_x'] + isizes['count_y']

dims_order = isizes.sort_values('total_count', ascending=False)[['imageDim']]

print('Number of Unique Resolutions Available are: ',len(dims_order))
isizes = pd.concat([train_isdf, test_isdf])

print('Number of Unique Resolutions Available in both train and test are',isizes.shape[0])
isizes.head()
popularSizes = isizes[isizes['fraction'] > 0.002]

popularSizes.shape
plt.figure(figsize=(20,14))

sns.barplot(x='imageDim',y='fraction',data = popularSizes, hue="set")

_ = plt.xticks(rotation=45)
def is_grey_scale(givenImage):

    """Adopted from https://www.kaggle.com/lextoumbourou/humpback-whale-id-data-and-aug-exploration"""

    w,h = givenImage.size

    for i in range(w):

        for j in range(h):

            r,g,b = givenImage.getpixel((i,j))

            if r != g != b: return False

    return True
sampleFrac = 0.1

#get our sampled images

imageList = [Image.open(f'{DIR}/train/{imageName}').convert('RGB')

            for imageName in train['Image'].sample(frac=sampleFrac)]
isGreyList = [is_grey_scale(givenImage) for givenImage in imageList]
#then get proportion greyscale

np.sum(isGreyList) / len(isGreyList)
sampleFrac = 0.1

imageListtest = [Image.open(f'{DIR}/test/{imageName}').convert('RGB')

            for imageName in test['Image'].sample(frac=sampleFrac)]

isGreyListtest = [is_grey_scale(givenImage) for givenImage in imageListtest]
#then get proportion greyscale

np.sum(isGreyListtest) / len(isGreyListtest)
def get_rgb_men(row):

    img = cv2.imread(DIR + '/train/' + row['Image'])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return np.sum(img[:,:,0]), np.sum(img[:,:,1]), np.sum(img[:,:,2])



train['R'], train['G'], train['B'] = zip(*train.apply(lambda row: get_rgb_men(row), axis=1) )
df = train[(train['B'] < train['R']) & (train['G'] < train['R'])]

num_photos = 6

fig, axr = plt.subplots(num_photos,2,figsize=(15,15))

for i,(_,row) in enumerate(df.iloc[:num_photos].iterrows()):

    img = cv2.imread(DIR + '/train/' + row['Image'])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axr[i,0].imshow(img)

    axr[i,0].axis('off')

    axr[i,1].set_title('R={:.0f}, G={:.0f}, B={:.0f} '.format(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))) 

    x, y = np.histogram(img[:,:,0], bins=255, normed=True)

    axr[i,1].bar(y[:-1], x, label='R', alpha=0.8, color='C0')

    x, y = np.histogram(img[:,:,1], bins=255, normed=True)

    axr[i,1].bar(y[:-1], x, label='G', alpha=0.8, color='C5')

    x, y = np.histogram(img[:,:,2], bins=255, normed=True)

    axr[i,1].bar(y[:-1], x, label='B', alpha=0.8, color='C1')

    axr[i,1].legend()

    axr[i,1].axis('off')
df = train[(train['B'] > train['R']) & (train['B'] > train['G'])]

num_photos = 6

fig, axr = plt.subplots(num_photos,2,figsize=(15,15))

for i,(_,row) in enumerate(df.iloc[:num_photos].iterrows()):

    img = cv2.imread(DIR + '/train/' + row['Image'])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axr[i,0].imshow(img)

    axr[i,0].axis('off')

    axr[i,1].set_title('R={:.0f}, G={:.0f}, B={:.0f} '.format(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))) 

    x, y = np.histogram(img[:,:,0], bins=255, normed=True)

    axr[i,1].bar(y[:-1], x, label='R', alpha=0.8, color='C0')

    x, y = np.histogram(img[:,:,1], bins=255, normed=True)

    axr[i,1].bar(y[:-1], x, label='G', alpha=0.8, color='C5')

    x, y = np.histogram(img[:,:,2], bins=255, normed=True)

    axr[i,1].bar(y[:-1], x, label='B', alpha=0.8, color='C1')

    axr[i,1].legend()

    axr[i,1].axis('off')
##Bounding Boxes for the whale fins only.

bbox = pd.read_csv('../input/bounding-box/bounding_boxes.csv')
bbox.head()
# DIR = '/home/aiml/ml/share/data/all_kagg'

TRAIN = os.path.join(DIR, 'train')

TEST = os.path.join(DIR, 'test')



train_paths = [img for img in os.listdir(TRAIN)]

test_paths = [img for img in os.listdir(TEST)]
len(train_paths)

len(test_paths)
## Create full path for the images

def full_path(row):

    if row in train_paths:

        return TRAIN+'/'+row

    else:

        return TEST+'/'+row
bbox['Full_Path'] = bbox['Image'].apply(lambda row: full_path(row))
##check images are already present in the directory or not.

bbox[bbox['Image'] ==test_paths[0]]
i=2

fig,ax = plt.subplots(6,2,figsize=(25,20))

for i in range(6):

    img_row = bbox[bbox['Image'] ==test_paths[i]]

    img = cv2.imread(TEST+'/'+img_row['Image'].values[0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # fig,ax = plt.subplots(2)

    ax[i,0].imshow(img)

    xmin1 = img_row['x0'].values[0]

    ymin1 = img_row['y0'].values[0]

    xmax = img_row['x1'].values[0]

    ymax = img_row['y1'].values[0]

    rect = patches.Rectangle((xmin1,ymin1),xmax-xmin1,ymax-ymin1,linewidth=1,edgecolor='r',facecolor='none')

    ax[i,1].add_patch(rect)

    ax[i,1].imshow(img)

    # plt.imshow(img)

plt.show()
def x_orig_img(row):

    if row in train_paths:

        return Image.open(TRAIN+'/'+row).size[0]

    else:

        return Image.open(TEST+'/'+row).size[0]



def y_orig_img(row):

    if row in train_paths:

        return Image.open(TRAIN+'/'+row).size[1]

    else:

        return Image.open(TEST+'/'+row).size[1]
bbox['x_orig'] = bbox['Image'].apply(lambda row: x_orig_img (row))

bbox['y_orig'] = bbox['Image'].apply(lambda row: y_orig_img (row))
bbox['ratio'] = ((bbox['x1']-bbox['x0']) * (bbox['y1']-bbox['y0']))/(bbox['x_orig'] * bbox['y_orig']) * 100
plt.figure(figsize=(20,14))

plt.title("Comparison Of Full and Cropped Images", {'size':'14'})

f = sns.distplot(bbox['ratio'])

f.set_xlabel("In Percentage Cropped Size Over Original", {'size':'14'})

f.set_ylabel("Frequency", {'size':'14'}) 
bbox[bbox['ratio']<5].sort_values(['ratio']).sort_values(['ratio']).head(5)
i=2

fig,ax = plt.subplots(6,2,figsize=(25,20))

for i in range(6):

    img_row = bbox[bbox['ratio']<5].sort_values(['ratio'],ascending=[False])[i:i+1]

    img = cv2.imread(img_row['Full_Path'].values[0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax[i,0].imshow(img)

    xmin1 = img_row['x0'].values[0]

    ymin1 = img_row['y0'].values[0]

    xmax = img_row['x1'].values[0]

    ymax = img_row['y1'].values[0]

    rect = patches.Rectangle((xmin1,ymin1),xmax-xmin1,ymax-ymin1,linewidth=1,edgecolor='r',facecolor='none')

    ax[i,1].add_patch(rect)

    ax[i,1].imshow(img)

    # plt.imshow(img)

plt.show()
bbox[bbox['ratio']>95].sort_values(['ratio']).sort_values(['ratio'],ascending=[False]).head()
i=2

fig,ax = plt.subplots(6,2,figsize=(25,20))

for i in range(6):

    img_row = bbox[bbox['ratio']>90].sort_values(['ratio'], ascending=[False])[i:i+1]

#     img_row = bbox[bbox['Image'] ==test_paths[i]]

    img = cv2.imread(img_row['Full_Path'].values[0])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax[i,0].imshow(img)

    xmin1 = img_row['x0'].values[0]

    ymin1 = img_row['y0'].values[0]

    xmax = img_row['x1'].values[0]

    ymax = img_row['y1'].values[0]

    rect = patches.Rectangle((xmin1,ymin1),xmax-xmin1,ymax-ymin1,linewidth=1,edgecolor='r',facecolor='none')

    ax[i,1].add_patch(rect)

    ax[i,1].imshow(img)

plt.show()
## I have used these awesome kernels for whole EDA

##https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric

##https://www.kaggle.com/artgor/pytorch-whale-identifier

##https://www.kaggle.com/kretes/eda-distributions-images-and-no-duplicates

##https://www.kaggle.com/cristianpb/on-finding-rgb-or-bgr

##https://www.kaggle.com/suicaokhoailang/generating-whale-bounding-boxes
##Model Building coming soon...