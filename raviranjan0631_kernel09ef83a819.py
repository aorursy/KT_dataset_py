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
from pathlib import Path

from keras.preprocessing import image

import matplotlib.pyplot as plt
# table to store the data

from prettytable import PrettyTable

table = PrettyTable()

table.field_names=["layers",'train accuracy', 'cv accuracy', 'test accuracy']

print(table)
p = Path('../input/dataset/dataset')

dirs = p.glob('*')

label_count = 0

label_dict = {}

# it will separate class names fom the dictionary and creates a dictionary with key respective to them

for folder in dirs:

    label = str(folder).split('/')[-1]

    label_dict[label] = label_count

    label_count += 1



print("There are",len(label_dict), "classes\n")





for x, y in label_dict.items():

  print(x, y)



image_data_per_class = []

image_labels_per_class = []

p = Path('../input/dataset/dataset')

dirs = p.glob('*')

total = 0

for folder in dirs:

    label = str(folder).split('/')[-1]

    cnt = 0

    img_data = []

    label_data = []

    print(label)

    # read png file 

    for img_path in folder.glob('*.png'):

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_data.append(img_array)

        cnt += 1

        total += 1

        label_data.append(label_dict[label])

    

    # read jpg files

    for img_path in folder.glob('*.jpg'):

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_data.append(img_array)

        cnt += 1

        total += 1

        label_data.append(label_dict[label])

        

    print("There are", cnt, "images in", label)

    image_data_per_class.append(img_data)

    image_labels_per_class.append(label_data)

print('There are' ,total,"images")
print(len(image_data_per_class[0]))

print(len(image_data_per_class[0][0]))

print(len(image_data_per_class[0][0][0]))

print(len(image_data_per_class[0][0][0][0]))
print(len(image_labels_per_class))
# method to draw the images

def drawImg(img, label):

    plt.imshow(img)

    for key, value in label_dict.items(): 

         if label == value: 

                plt.title(key)

    plt.show()
# Visualization

import numpy as np

for i in range(0,149):

    x = np.array(image_data_per_class[i])

    y = np.array(image_labels_per_class[i])

    drawImg(x[0]/255.0, y[0])
plt.plot( [x for x in range(0,149)], [len(y) for y in image_labels_per_class],'*')

plt.grid()

plt.xlabel('classes')

plt.ylabel('number of elements')

plt.show()
#get the class label with maximum data points

max = -1

label = -1

for i in range(0,149):

    if max < len(image_data_per_class[i]):

        max = len(image_data_per_class[i])

        label = i

print("max number of data points is", max, 'and label is', label)
from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(rotation_range = 10, width_shift_range=0.1,

                        height_shift_range = 0.1, shear_range = 0.15, zoom_range = 0.1, 

                        channel_shift_range=10, horizontal_flip = True)
image_data_per_class = []

image_labels_per_class = []

p = Path('../input/dataset/dataset')

dirs = p.glob('*')

total = 0

for folder in dirs:

    label = str(folder).split('/')[-1]

    cnt = 0

    size = 303

    img_data = []

    label_data = []

    img = []

    for img_path in folder.glob('*.png'):

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_data.append(img_array)

        cnt += 1

        total += 1

        label_data.append(label_dict[label])

    

    

    for img_path in folder.glob('*.jpg'):

        img = image.load_img(img_path, target_size=(150, 150))

        img_array = image.img_to_array(img)

        img_data.append(img_array)

        cnt += 1

        total += 1

        label_data.append(label_dict[label])

    

    if size - cnt > 0:

        aug_iter = gen.flow(np.expand_dims(img,0))

        aug_samples = [next(aug_iter)[0].astype(np.uint8) for i in range(303-cnt)]

        for i in range (303-cnt):

            label_data.append(label_dict[label])

        total += len(aug_samples)

        for sample in aug_samples:

            img_array = image.img_to_array(sample)

            img_data.append(img_array)

            

    image_data_per_class.append(img_data)

    image_labels_per_class.append(label_data)

print('There are' ,total,"images")
plt.plot( [x for x in range(0,149)], [len(y) for y in image_labels_per_class],'*')

plt.grid()

plt.xlabel('classes')

plt.ylabel('number of elements')

plt.show()