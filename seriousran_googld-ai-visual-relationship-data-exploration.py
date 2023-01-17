import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import cv2



import os

print(os.listdir("../input"))

print(os.listdir("../input/vrd01"))

print(os.listdir("../input/open-images-2019-visual-relationship"))
df_train = pd.read_csv('../input/vrd01/challenge-2018-train-vrd.csv')

print('shape of train data frame:', df_train.shape)

df_sample = pd.read_csv('../input/open-images-2019-visual-relationship/VRD_sample_submission.csv')

print('shape of sample submission data frame:', df_sample.shape)
df_train.head()
df_train['RelationshipLabel'].value_counts()
numerical = ['XMin1', 'XMax1', 'YMin1', 'YMax1', 'XMin2', 'XMax2', 'YMin2', 'YMax2']
df_train[numerical].hist(bins=15, figsize=(20, 10), layout=(2, 4));
df_sample.head()
values_what = df_sample[df_sample['ImageId']=='b4c3b52a8723d431']['PredictionString'].values

values = str(values_what)[2:-2].split(' ')



print('confidence: ', values[0])



print('label 1: ', values[1])

print('XMin1: ', values[2])

print('YMin1: ', values[3])

print('XMax1: ', values[4])

print('YMax1: ', values[5])

print('Label 2: ', values[6])

print('XMin2: ', values[7])

print('YMin2: ', values[8])

print('XMax2: ', values[9])

print('YMax2: ', values[10])

print('Relation Label: ', values[11])
print('confidence: ', values[12])

print('label 1: ', values[13])

print('XMin1: ', values[14])

print('YMin1: ', values[15])

print('XMax1: ', values[16])

print('YMax1: ', values[17])

print('Label 2: ', values[18])

print('XMin2: ', values[19])

print('YMin2: ', values[20])

print('XMax2: ', values[21])

print('YMax2: ', values[22])

print('Relation Label: ', values[23])
image_filenames = os.listdir("../input/open-images-2019-visual-relationship/test")



import random

for i in range(10):

    index = random.randrange(len(image_filenames))

    path = "../input/open-images-2019-visual-relationship/test" + "/" + image_filenames[index]

    src_img = cv2.imread(path)

    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))

    plt.show()