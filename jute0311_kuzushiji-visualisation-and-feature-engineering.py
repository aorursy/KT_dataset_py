from PIL import Image, ImageDraw, ImageFont

from os import listdir

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data
fontsize = 50



font = ImageFont.truetype('../input/font-data/NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
df_train = pd.read_csv('../input/kuzushiji-recognition/train.csv')

unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/kuzushiji-recognition/unicode_translation.csv').values}

unicode_count = {codepoint: 0 for codepoint, _ in pd.read_csv('../input/kuzushiji-recognition/unicode_translation.csv').values}

box_hw_count = unicode_count.copy()

box_categorize = unicode_count.copy()
# make label for prediction.

unicode_label = {count:codeprint for count,(codeprint,_) in enumerate(unicode_map.items())}
# This function takes in a filename of an image, and the labels in the string format given in train.csv, and returns an image containing the bounding boxes and characters annotated

def visualize_training_data(image_fn, labels):

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 5)

    

    # Read image

    imsource = Image.open(image_fn).convert('RGBA')

    bbox_canvas = Image.new('RGBA', imsource.size)

    char_canvas = Image.new('RGBA', imsource.size)

    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character

    char_draw = ImageDraw.Draw(char_canvas)



    for codepoint, x, y, w, h in labels:

        x, y, w, h = int(x), int(y), int(w), int(h)

        char = unicode_map[codepoint] # Convert codepoint to actual unicode character



        # Draw bounding box around character, and unicode character next to it

        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))

        char_draw.text((x + w + fontsize/4, y + h/2 - fontsize), char, fill=(0, 0, 255, 255), font=font)



    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)

    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.

    return np.asarray(imsource)
np.random.seed(1337)



for i in range(2):

    img, labels = df_train.values[np.random.randint(len(df_train))]

    viz = visualize_training_data('../input/kuzushiji-recognition/train_images/{}.jpg'.format(img), labels)

    

    plt.figure(figsize=(15, 15))

    plt.title(img)

    plt.imshow(viz, interpolation='lanczos')

    plt.show()
for i in range(len(df_train)):

    img, labels = df_train.values[i]

    if type(labels) == float:

        continue

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 5)

    

    for codepoint, x, y, w, h in labels:

        unicode_count[codepoint] += 1
unicode_count_df = pd.io.json.json_normalize(unicode_count)

unicode_count_dict ={

    "Unicode": unicode_count_df.columns.tolist(),

    "char":list(unicode_map.values()),

    "count": list(unicode_count_df.values[0])

}

unicode_count_df = pd.DataFrame(unicode_count_dict)
characters_sorted_by_count = unicode_count_df.query('count > 0').sort_values('count', ascending=False)

characters_sorted_by_count
characters_categorize = pd.DataFrame(

    {

        'number of appearances':['0','1~10','11~50','51~100','101~500','501~1000','1001~5,000','5,001~10,000','10,001~','1~','0~'],

        'count':[len(unicode_count_df.query('count==0')),

                 len(unicode_count_df.query('1<=count<=10')),

                 len(unicode_count_df.query('11<=count<=50')),

                len(unicode_count_df.query('51<=count<=100')),

                len(unicode_count_df.query('101<=count<=500')),

                len(unicode_count_df.query('501<=count<=1000')),

                len(unicode_count_df.query('1001<=count<=5000')),

                len(unicode_count_df.query('5001<=count<=10000')),

                len(unicode_count_df.query('10001<=count')),

                len(unicode_count_df.query('1<=count')),

                len(unicode_count_df),]

    }

)
characters_categorize
#  character list

print(unicode_count_df.query('count==0')["char"].tolist())
print(unicode_count_df.query('1<=count<=10')["char"].tolist())
print(unicode_count_df.query('11<=count<=50')["char"].tolist())
print(unicode_count_df.query('51<=count<=100')["char"].tolist())
print(unicode_count_df.query('101<=count<=500')["char"].tolist())
print(unicode_count_df.query('501<=count<=1000')["char"].tolist())
print(unicode_count_df.query('1001<=count<=5000')["char"].tolist())
print(unicode_count_df.query('5001<=count<=10000')["char"].tolist())
print(unicode_count_df.query('10001<=count')["char"].tolist())
for i in range(len(df_train)):

    img, labels = df_train.values[i]

    if type(labels) == float:

        continue

    # Convert annotation string to array

    labels = np.array(labels.split(' ')).reshape(-1, 5)

    

    for codepoint, x, y, w, h in labels:

        if h < w:

            box_hw_count[codepoint] += 1
box_info_dict = unicode_count_dict.copy()

box_info_dict["h<w"] = [box_hw_count[codeprint] for codeprint in unicode_count_dict["Unicode"]]



box_info = pd.DataFrame(box_info_dict)



box_info
box_rate =  box_info['h<w'] / box_info['count']

box_info = pd.concat([box_info,box_rate],axis=1)

box_info.columns = ['Unicode','char','count','h<w','rate']

box_info
for i in range(len(box_info)):

    if box_info.iat[i,4] >= 0.90:

        box_categorize[box_info.iat[i,0]] = 1

    elif box_info.iat[i,4] <= 0.10:

        box_categorize[box_info.iat[i,0]] = 3

    elif 0.10 < box_info.iat[i,4] < 0.90:

        box_categorize[box_info.iat[i,0]] = 2

    else :

        box_categorize[box_info.iat[i,0]] = 0

    

box_categorize_df = pd.DataFrame({'categorize':list(box_categorize.values())})

box_info = pd.concat([box_info,box_categorize_df],axis=1)



box_info
box_info_restricted = box_info.query('count>100')
box_categorize_distribution = pd.DataFrame({

    'category':[0,1,2,3],

    'count':[

        len(box_info.query('categorize == 0')),

        len(box_info.query('categorize == 1')),

        len(box_info.query('categorize == 2')),

        len(box_info.query('categorize == 3')),

    ]

})
box_categorize_distribution
box_categorize_distribution_ristricted = pd.DataFrame({

    'category':[0,1,2,3,4],

    'count':[

        len(box_info_restricted.query('categorize == 0')),

        len(box_info_restricted.query('categorize == 1')),

        len(box_info_restricted.query('categorize == 2')),

        len(box_info_restricted.query('categorize == 3')),

        len(box_info_restricted.query('categorize == 4')),

    ]

})
box_categorize_distribution_ristricted
box_info_restricted.query('categorize == 1')
box_info_restricted.query('categorize == 2')
box_info_restricted.query('categorize == 3')