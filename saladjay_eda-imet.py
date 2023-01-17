# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import gc

import os

import PIL

import glob

import cv2

import time



from scipy import stats

from multiprocessing import Pool

from PIL import ImageOps,ImageFilter,Image

from tqdm import tqdm

from wordcloud import WordCloud



tqdm.pandas()
train_size_df = pd.read_csv('../input/train.csv')
width = []

height = []

for name in tqdm(train_size_df['id']):

    img = Image.open('../input/train/'+name+'.png')

    width.append(img.size[0])

    height.append(img.size[1])

train_size_df['width'] = width

train_size_df['height'] = height
train_size_df.head(5)
width_height_ratio = np.zeros(len(train_size_df))

height_width_ratio = np.zeros(len(train_size_df))

for row_index,(w,h) in tqdm(enumerate(zip(train_size_df['width'],train_size_df['height']))):

    if w==300:

        times = h/w

        times = round(times,1)

        height_width_ratio[row_index] = times

    else:

        times = w/h

        times = round(times,1)

        width_height_ratio[row_index] = times

train_size_df['width_height_ratio'] = width_height_ratio

train_size_df['height_width_ratio'] = height_width_ratio
train_size_df['ratio'] = train_size_df['height_width_ratio'] + train_size_df['width_height_ratio']
train_size_df['ratio'].max(),train_size_df['ratio'].min(),train_size_df['ratio'].mean(),train_size_df['ratio'].quantile(q=0.25),train_size_df['ratio'].median(),train_size_df['ratio'].quantile(q=0.75)
ratio_1_2 = train_size_df[train_size_df['ratio']<=2]

print(ratio_1_2.shape)

plt.figure(figsize=(20,8))

ax = sns.countplot(ratio_1_2['ratio'])

plt.xlabel('ratio_1_2')

plt.title('Number of image per ratio', fontsize=20)



for p in ax.patches:

    ax.annotate(f'{p.get_height()}-{p.get_height() * 100 / ratio_1_2.shape[0]:.3f}%',

            (p.get_x() + p.get_width() / 2., p.get_height()), 

            ha='center', 

            va='center', 

            fontsize=11, 

            color='black',

            xytext=(0,7), 

            textcoords='offset points')
ratio_section = [1.0,1.3,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,20.0,26.0]

ratio_section_count = np.zeros(len(ratio_section)-1)

w_ratio_section_count = np.zeros(len(ratio_section)-1)

h_ratio_section_count = np.zeros(len(ratio_section)-1)

for i in range(len(ratio_section_count)):

    start = ratio_section[i]

    end = ratio_section[i+1]-0.1

    ratio_section_count[i] = train_size_df['ratio'].between(start,end).sum()

    w_ratio_section_count[i] = train_size_df['width_height_ratio'].between(start,end).sum()

    h_ratio_section_count[i] = train_size_df['height_width_ratio'].between(start,end).sum()

ratio_section_count = ratio_section_count.astype(np.int64)

w_ratio_section_count = w_ratio_section_count.astype(np.int64)

h_ratio_section_count = h_ratio_section_count.astype(np.int64)
def draw(start,end,column='ratio'):

    temp_df = train_size_df[train_size_df[column].between(start,end)]

    temp_labels_count = np.zeros((len(temp_df),1103))

    num_temp_labels = np.zeros(len(temp_df))

    for row_index,row in enumerate(temp_df['attribute_ids']):

        ids = row.split(' ')

        num_temp_labels[row_index] = len(ids)

        for id_index in ids:

            temp_labels_count[row_index,int(id_index)] = 1

    label_sum = np.sum(temp_labels_count, axis=0)

    attributes_sequence = label_sum.argsort()[::-1]

    label_names = pd.read_csv('../input/labels.csv')

    label_names = label_names['attribute_name']

    attributes_labels = [label_names[x] for x in attributes_sequence]

    attributes_counts = [label_sum[x] for x in attributes_sequence]

    plt.figure(figsize=(20,2))



    plt.subplot()

    ax1 = sns.barplot(y=attributes_labels[:5], x=attributes_counts[:5], orient="h")

    plt.title(f'Label Counts between {start} and {end} (Top 5)',fontsize=15)

    plt.xlim((0, max(attributes_counts)*1.15))

    plt.yticks(fontsize=15)



    for p in ax1.patches:

        ax1.annotate(f'{int(p.get_width())}-{p.get_width() * 100 / temp_df.shape[0]:.2f}%',

                    (p.get_width(), p.get_y() + p.get_height() / 2.), 

                    ha='left', 

                    va='center', 

                    fontsize=10, 

                    color='black',

                    xytext=(7,0), 

                    textcoords='offset points')

    plt.show()
for ratio in [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2]:

    draw(ratio,ratio)
for ratio in [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2]:

    draw(ratio,ratio,'height_width_ratio')
for ratio in [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2]:

    draw(ratio,ratio,'width_height_ratio')
train_size_df.to_csv('train_size_df.csv')
train_hsv_df = pd.read_csv('../input/train.csv')
gray_img = np.zeros(len(train_hsv_df))

h_list = np.zeros(len(train_hsv_df))

s_list = np.zeros(len(train_hsv_df))

v_list = np.zeros(len(train_hsv_df))

for row,img_name in tqdm(enumerate(train_hsv_df['id'])):

    img = cv2.imread('../input/train/'+img_name+'.png')

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h,s,v = np.average(hsv_img,axis=(0,1))

    if h == 0:

        gray_img[row] = 1

    h_list[row] = h

    s_list[row] = s

    v_list[row] = v

train_hsv_df['gray_img'] = gray_img

train_hsv_df['h'] = h_list

train_hsv_df['s'] = s_list

train_hsv_df['v'] = v_list
train_hsv_df.to_csv('train_hsv_df.csv')
gray_df = train_hsv_df[train_hsv_df['gray_img']==1]

print(gray_df.shape)
gray_labels = np.zeros((len(gray_df),1103))

for row_index,row in enumerate(gray_df['attribute_ids']):

    for label in row.split(' '):

        gray_labels[row_index,int(label)] = 1

train_labels = np.zeros((len(train_hsv_df),1103))

for row_index,row in enumerate(train_hsv_df['attribute_ids']):

    for label in row.split(' '):

        train_labels[row_index,int(label)] = 1
gray_sums = gray_labels.sum(axis=0)

train_sums = train_labels.sum(axis=0)

percentage = gray_sums/train_sums



label_names = pd.read_csv('../input/labels.csv')

label_names = label_names['attribute_name']

new_img_df = pd.DataFrame(index=label_names)

new_img_df['gray_count'] = gray_sums

new_img_df['count'] = train_sums

new_img_df['percentage'] = percentage

new_img_df.head(5)
new_img_df.to_csv('new_img_df.csv')
new_img_df.sort_values(['percentage'],ascending=False).iloc[:30]
new_img_df.sort_values(['gray_count','percentage'],ascending=False).iloc[:30]
markets = []

markets_index = []

for i,name in enumerate(label_names):

    if name.endswith('market'):

        print(name)

        markets.append(name)

        markets_index.append(i)
new_img_df.loc[markets]
markets_index
for i in np.random.choice(np.where(train_labels[:,134]==1)[0],3):

    img_path = '../input/train/' + train_hsv_df.iloc[i]['id'] + '.png'

    img = Image.open(img_path)

    plt.imshow(img)

    plt.show()

    print([label_names[int(i)] for i in train_hsv_df.iloc[i]['attribute_ids'].split(' ')])
for i in np.random.choice(np.where(train_labels[:,135]==1)[0],3):

    img_path = '../input/train/' + train_hsv_df.iloc[i]['id'] + '.png'

    img = Image.open(img_path)

    plt.imshow(img)

    plt.show()

    print([label_names[int(i)] for i in train_hsv_df.iloc[i]['attribute_ids'].split(' ')])
for i in np.random.choice(np.where(train_labels[:,142]==1)[0],2):

    img_path = '../input/train/' + train_hsv_df.iloc[i]['id'] + '.png'

    img = Image.open(img_path)

    plt.imshow(img)

    plt.show()

    print([label_names[int(i)] for i in train_hsv_df.iloc[i]['attribute_ids'].split(' ')])
def mask2(img_path):

    image=cv2.imread(img_path)

    img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_image2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h,s,v = cv2.split(hsv_image)

    lower_blue=np.array([20,30,0])

    upper_blue=np.array([160,255,255])

    lower_blue2=np.array([0,30,0])

    upper_blue2=np.array([255,255,255])

    mask=cv2.inRange(hsv_image,lower_blue,upper_blue)

    mask2=cv2.inRange(hsv_image2,lower_blue2,upper_blue2)

    use_mask = False

    if np.average(h)<20 or np.average(s)<30 or np.average(v)<140:

        res = img

        res2 = cv2.bitwise_and(img,img,mask=mask)

        res3 = cv2.bitwise_and(img,img,mask=mask2)

        res4 = (res2+res3)//2

        res4[np.where(res4>255)]=255

    else:

        res = cv2.bitwise_and(img,img,mask=mask)

        res2 = cv2.bitwise_and(img,img,mask=mask)

        res3 = cv2.bitwise_and(img,img,mask=mask2)

        res4 = (res2+res3)//2

        res4[np.where(res4>255)]=255

        use_mask = True

    plt.subplot(171),plt.imshow(img),plt.title('ORIGINAL')

    plt.subplot(172),plt.imshow(mask),plt.title('Mask1')

    plt.subplot(173),plt.imshow(mask2),plt.title('Mask2')

    plt.subplot(174),plt.imshow(res),plt.title('use_mask' if use_mask else 'original')

    plt.subplot(175),plt.imshow(res2),plt.title('use_mask_BGR')

    plt.subplot(176),plt.imshow(res3),plt.title('use_mask_RGB')

    plt.subplot(177),plt.imshow(res4),plt.title('res4')

    plt.show()

    h,s,v=np.average(hsv_image,axis=(0,1))

    print(h, s, v)

    h2,s2,v2=np.average(hsv_image2,axis=(0,1))

    print(h2, s2, v2)

    if h == 0 and s == 0:

        print(np.average(img,axis=(0,1)))
for i in np.random.choice(range(len(train_hsv_df)),3):

    plt.figure(figsize=(20,20))

    img_id = train_hsv_df.iloc[i]['id']

    path = '../input/train/'+img_id+'.png'

    mask2(path)