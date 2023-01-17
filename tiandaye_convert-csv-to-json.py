import cv2

import os

import pandas as pd

from tqdm import tqdm

import json

import base64
img_pth = '../input/global-wheat-detection/train/'

pattern = '../input/hourse/hourse1.json'

train_csv = pd.read_csv('../input/global-wheat-detection/train.csv');train_csv.head()
train_csv['bbox'] = train_csv['bbox'].apply(lambda x: x[1:-1].split(","))

train_csv['x'] = train_csv['bbox'].apply(lambda x: x[0]).astype('float32')

train_csv['y'] = train_csv['bbox'].apply(lambda x: x[1]).astype('float32')

train_csv['w'] = train_csv['bbox'].apply(lambda x: x[2]).astype('float32')

train_csv['h'] = train_csv['bbox'].apply(lambda x: x[3]).astype('float32')



train_csv['w'] = train_csv['x'] + train_csv['w']

train_csv['h'] = train_csv['y'] + train_csv['h']

train_csv = train_csv.rename(columns={'x': 'x1', 'y': 'y1', 'w': 'x2', 'h': 'y2'})
train_csv.head()
def readJson(pth):

    f = open(pth, encoding='utf-8')

    file = json.load(f)

    return file
for img_name in tqdm(os.listdir(img_pth)):

    json_pattern = readJson(pattern)

    #print(img_name)  # 00333207f.jpg

    img_id = img_name.split('.')[0]

    annos = train_csv.loc[train_csv['image_id'] == img_id]

    #break

    json_pattern['imagePath'] = img_name

    json_pattern['imageHeight'] = 1024

    json_pattern['imageWidth'] = 1024

    json_pattern['imageData'] = base64.b64encode(open(img_pth + img_name, "rb").read()).strip().decode()

    

    json_pattern['shapes'] = []

    n = len(annos)

    for i in range(n):

        anno = annos.iloc[i]

        item = {}

        item['label'] = 'wheat'

        item['points'] = [[anno['x1']*1., anno['y1']*1.], [anno['x2']*1., anno['y2']*1.]]

        item['group_id'] = None

        item['shape_type'] = 'rectangle'

        item['flags'] = dict()

        json_pattern['shapes'].append(item)

        

    with open("../working/" + img_id + '.json', 'w') as f:

        f.write(json.dumps(json_pattern))