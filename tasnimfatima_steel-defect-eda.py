# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2

%matplotlib inline

import matplotlib.pyplot as plt

from fractions import Fraction

# Any results you write to the current directory are saved as output.
path =  '../input/severstal-steel-defect-detection/'
train = pd.read_csv(path + 'train.csv')

pd.set_option('display.max_colwidth',-1)

print(train.head(5))
df = train[train['EncodedPixels'].notnull()]

df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

class1_data = df[df['ClassId']=='1']

class2_data = df[df['ClassId']=='2']

class3_data = df[df['ClassId']=='3']

class4_data = df[df['ClassId']=='4']

df.head()

def find_percentage(a,b):

    return round((a/b)*100, 3)
print("Total Images: ", train.shape[0])

print("Images with Defects: ", df.shape[0])

print("Defected Image Percentage: ", find_percentage(df.shape[0], train.shape[0]))

print("Images with Defect class 1 ", class1_data.shape[0], " and percentage: ", find_percentage(class1_data.shape[0], train.shape[0]))

print("Images with Defect class 2 ", class2_data.shape[0], " and percentage: ", find_percentage(class2_data.shape[0], train.shape[0]))

print("Images with Defect class 3 ", class3_data.shape[0], " and percentage: ", find_percentage(class3_data.shape[0], train.shape[0]))

print("Images with Defect class 4 ", class4_data.shape[0], " and percentage: ", find_percentage(class4_data.shape[0], train.shape[0]))
def rle2mask(mask_rle, shape=(1600,256)):

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
columns = 2

rows = 10

fig = plt.figure(figsize=(40,columns*rows+2))

fn = class1_data['ImageId'].iloc[0]

c = class1_data['ClassId'].iloc[0]

#fig.add_subplot(rows, columns, i).set_title(fn+"  ClassId="+c)

pth = "../input/severstal-steel-defect-detection/train_images/" + fn

print(pth)

img = cv2.imread(pth)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = rle2mask(class1_data['EncodedPixels'].iloc[0])

image = img

plt.imshow(image)

plt.show()

fig = plt.figure(figsize=(40,columns*rows+2))

img[mask==1,0] = 255

print("Class 1 Defect")

plt.imshow(img)

plt.show()
columns = 2

rows = 10

fig = plt.figure(figsize=(40,columns*rows+2))

fn = class2_data['ImageId'].iloc[0]

c = class2_data['ClassId'].iloc[0]

#fig.add_subplot(rows, columns, i).set_title(fn+"  ClassId="+c)

pth = "../input/severstal-steel-defect-detection/train_images/" + fn

print(pth)

img = cv2.imread(pth)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = rle2mask(class2_data['EncodedPixels'].iloc[0])

image = img

plt.imshow(image)

plt.show()

fig = plt.figure(figsize=(40,columns*rows+2))

img[mask==1,0] = 34

print("Class 2 Defect ")

plt.imshow(img)

plt.show()
columns = 2

rows = 10

fig = plt.figure(figsize=(40,columns*rows+2))

fn = class3_data['ImageId'].iloc[0]

c = class3_data['ClassId'].iloc[0]

#fig.add_subplot(rows, columns, i).set_title(fn+"  ClassId="+c)

pth = "../input/severstal-steel-defect-detection/train_images/" + fn

print(pth)

img = cv2.imread(pth)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = rle2mask(class3_data['EncodedPixels'].iloc[0])

image = img

plt.imshow(image)

plt.show()

fig = plt.figure(figsize=(40,columns*rows+2))

img[mask==1,1] = 204

print("Class 3 Defect ")

plt.imshow(img)

plt.show()
columns = 2

rows = 10

fig = plt.figure(figsize=(40,columns*rows+2))

fn = class4_data['ImageId'].iloc[0]

c = class4_data['ClassId'].iloc[0]

#fig.add_subplot(rows, columns, i).set_title(fn+"  ClassId="+c)

pth = "../input/severstal-steel-defect-detection/train_images/" + fn

print(pth)

img = cv2.imread(pth)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = rle2mask(class4_data['EncodedPixels'].iloc[0])

image = img

plt.imshow(image)

plt.show()

fig = plt.figure(figsize=(40,columns*rows+2))

img[mask==1,1] = 0

print("Class 4 Defect ")

plt.imshow(img)

plt.show()
class1per = ()