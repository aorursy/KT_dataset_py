import numpy as np
import pandas as pd
from os import listdir
import os
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Image
files_location = '../input/generated-car-ocr/generated_60k'
files = listdir(files_location)
files[:5]
def renameFiles():
    vocab = {'а' : 'a', 'в':'b', 'е':'e', 'к': 'k', 'м':'m','н':'h', 'о':'o', 'р':'p', 'с':'c', 'т':'t', 'у':'y', 'х':'x'}
    for file in tqdm(files):
        fullPath = os.path.join(files_location, file)
        newFile = file
        for symbol in file:
            if symbol in vocab:
                newFile = newFile.replace(symbol, vocab[symbol])
        os.rename(fullPath, os.path.join(files_location, np))
# renameFiles()
# files = listdir(files_location)
info = defaultdict(lambda : [])
failures = []
for file in tqdm(files):
    fullPath = os.path.join(files_location, file)
    try:
        image = cv2.imdecode(np.fromfile(fullPath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except:
        failures.append(file)
        continue
    if image is None:
        failures.append(file)
        continue
    info['file'].append(file)
    number = file.split('.', 1)[0]
    for i in range(0,6):
        info[f'symbol{i + 1}'].append(number[i])
    region = number[6:]
    info['region'].append(int(region))
    info['region_length'].append(len(region))
    for i in range(0, 3):
        if i < len(region):
            info[f'region{i + 1}'].append(region[i])
        else:
            info[f'region{i + 1}'].append(' ')
    info['height'].append(image.shape[0])
    info['width'].append(image.shape[1])
    info['ratio'].append(image.shape[0]/image.shape[1])
print(len(failures))
Image(filename=os.path.join(files_location, failures[144]))
dataFrame = pd.DataFrame(info)
dataFrame
figure, axes = plt.subplots(3, 1,figsize=(10,13))

for index, name in enumerate(['ratio', 'width', 'height']):
    axes[index].set_title(name)
    dataFrame[name].hist(ax = axes[index])
figure, axes = plt.subplots(4, 1,figsize=(10,23))

dataFrame['region_length'].hist(ax = axes[0])
axes[0].set_title('Region length')

dataFrame['region1'].value_counts().plot(kind='bar', ax=axes[1], title = 'First region symbol')
dataFrame['region2'].value_counts().plot(kind='bar', ax=axes[2], title = 'Second region symbol')
dataFrame['region3'][dataFrame['region3'] != ' '].value_counts().plot(kind='bar', ax=axes[3], title = 'Third region symbol ')
figure, axes = plt.subplots(3, 1,figsize=(10,13))

for index, name in enumerate(['symbol1', 'symbol5', 'symbol6']):
    axes[index].set_title(name)
    dataFrame[name].value_counts().plot(kind='bar', ax = axes[index])
figure, axes = plt.subplots(3, 1,figsize=(10,13))

for index, name in enumerate(['symbol2', 'symbol3', 'symbol4']):
    axes[index].set_title(name)
    dataFrame[name].value_counts().plot(kind='bar', ax = axes[index])