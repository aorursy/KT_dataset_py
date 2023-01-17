import numpy as np
import pandas as pd
from os import listdir
import os
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import json
import math
from tqdm import tqdm
dataFolder = '../input/made-cv2/data'
with open(os.path.join(dataFolder, 'train.json')) as json_file:
    train_data = json.load(json_file)
vocabulary = defaultdict(lambda: 0)
for entry in tqdm(train_data):
    for number in entry['nums']:
        text = number['text'].upper()
        
        for s in text:
            vocabulary[s] +=1
series = pd.Series(vocabulary)
series[[a for a in series.index if ord(a) <= 57]].plot(kind='bar')
onlyLetters = series[[a for a in series.index if ord(a) > 57]]
onlyLetters.plot(kind='bar')
def getSymbol(text, index):
    replacements = {'В': 'B', 'Е': 'E', 'С':'C', 'Х':'X', 'А':'A', 'К':'K', 'М':'M', 'Н': 'H', 'О':'O', 'Р' :'P','Т':'T', 'У':'Y'}
    if text[index] in replacements:
        return replacements[text[index]]
    
    return text[index]
def draw(dataFrame):
    if dataFrame.shape[0] == 0:
        print('empty')
        return
    
    fig, ax = plt.subplots(nrows=dataFrame.shape[0], ncols=1, figsize=(13,dataFrame.shape[0] * 13))
    for idx, (index, item) in enumerate(dataFrame.iterrows()):
        image = cv2.imread(os.path.join(dataFolder, item['file']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = item['bbox']
        image = cv2.line(image, (bbox[0][0],bbox[0][1]), (bbox[1][0],bbox[1][1]), (128, 0, 128), 5)
        image = cv2.line(image, (bbox[1][0],bbox[1][1]), (bbox[2][0],bbox[2][1]), (128, 0, 0), 5)
        image = cv2.line(image, (bbox[2][0],bbox[2][1]), (bbox[3][0],bbox[3][1]), (0, 128, 0), 5)
        image = cv2.line(image, (bbox[3][0],bbox[3][1]), (bbox[0][0],bbox[0][1]), (0, 0, 128), 5)
        
        
        if dataFrame.shape[0] == 1:
            current_ax = ax
        else:
            current_ax = ax[idx]
        current_ax.imshow(image)
        current_ax.set_title(item['file'] + str(item['angle']))

    plt.show()
def loadNumber(path, bbox):
    image = cv2.imread(os.path.join(dataFolder, path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    minX = max(min([point[0] for point in bbox]), 0)
    maxX = max([point[0] for point in bbox])
    minY = max(min([point[1] for point in bbox]), 0)
    maxY = max([point[1] for point in bbox])
    if image is None:
        print(path)
    crop_img = image[minY:maxY, minX:maxX]

    return crop_img
info = defaultdict(lambda : [])

for entry in tqdm(train_data):
    for number in entry['nums']:
        text = number['text'].upper()
        if entry['file'] =='train/25632.bmp':
            continue
            
        for i in range(0,9):
            symbol = ' '
            if i < len(text):
                symbol = getSymbol(text, i)

            info[f'symbol{i + 1}'].append(symbol)
        
        bbox = number['box']
        topLeft = bbox[0]
        topRight = bbox[1]
        botRight = bbox[2]
        botLeft = bbox[3]

        width = math.hypot(topLeft[0] - topRight[0], topLeft[1] - topRight[1])
        height = math.hypot(topLeft[0] - botLeft[0], topLeft[1] - botLeft[1])
        angle = np.rad2deg(np.arctan2(topLeft[1] - topRight[1], topRight[0] - topLeft[0]))
        square = width * height
        info['file'].append(entry['file'])
        info['width'].append(width)
        info['height'].append(height)
        info['ratio'].append(width / height)
        info['angle'].append(angle)
        info['square'].append(square)
        info['bbox'].append(bbox)
        info['length'].append(len(number['text']))
dataFrame = pd.DataFrame(info)
draw(dataFrame.nlargest(10, ['angle']))
draw(dataFrame.nsmallest(10, ['angle']))
draw(dataFrame.nsmallest(10, ['square']))
draw(dataFrame.nlargest(10, ['square']))
draw(dataFrame.nsmallest(10, ['ratio']))
draw(dataFrame.nlargest(10, ['ratio']))
dataFrame['length'].hist()
figure, axes = plt.subplots(9, 1,figsize=(10,33))

for i in range(0, 9):
    name = f'symbol{i+1}'
    dataFrame[name][dataFrame[name] != ' '].value_counts().plot(kind='bar', ax=axes[i], title = f'{i+1} symbol ')
draw(dataFrame[dataFrame['symbol1'] == '9'].head())
draw(dataFrame[dataFrame['symbol1'] == '0'].head())
draw(dataFrame[dataFrame['symbol2'] == 'M'].head())
draw(dataFrame[dataFrame['symbol4'] == 'A'].head())
